import sys

import yaml
import os

import jax
import jax.numpy as jnp
import optax
from flax import serialization, traverse_util
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import wandb
import numpy as np
from PIL import Image

from model import AutoencoderDC
from discriminator import Discriminator
from lpips import PerceptualLoss


def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    print(f"Loading checkpoint from {path}...")
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def adopt_weight(step, threshold, value=0.0):
    return jnp.where(step < threshold, value, 1.0)


def ema_update(ema_params, new_params, decay):
    return jax.tree_util.tree_map(
        lambda e, p: decay * e + (1.0 - decay) * p,
        ema_params,
        new_params
    )

def create_optimizer(learning_rate, weight_decay, beta1=0.9, beta2=0.999):
    def weight_decay_mask(params):
        return jax.tree_util.tree_map(lambda x: x.ndim > 1, params)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
            mask=weight_decay_mask
        )
    )


def create_phase2_optimizer(params, learning_rate=1.6e-5):
    partition_optimizers = {
        'trainable': optax.adamw(learning_rate=learning_rate, b1=0.9, b2=0.999, weight_decay=0.001),
        'frozen': optax.set_to_zero()
    }

    def map_fn(path, v):
        path_str = '/'.join(path)

        if 'Encoder_0/SanaMultiscaleLinearAttention_5' in path_str or 'Encoder_0/GLUMBConv_5' in path_str or 'Encoder_0/Conv_0' in path_str:
            return 'trainable'

        if 'Decoder_0/Conv_0' in path_str or 'Decoder_0/DCUpBlock2d_0' in path_str or 'norm' in path_str:
            return 'trainable'

        return 'frozen'

    param_partitions = traverse_util.path_aware_map(map_fn, params)

    return optax.multi_transform(partition_optimizers, param_partitions)


def make_generator_update_fn(
        *,
        dcae_apply_fn,
        dcae_optimizer,
        disc_apply_fn,
        disc_optimizer,
        perceptual_apply_fn,
        reconstruction_weight,
        perceptual_weight,
        generator_weight,
        discriminator_weight,
        disc_start,
        ema_decay,
):
    def update_fn(
            dcae_params,
            dcae_opt_state,
            disc_params,
            disc_opt_state,
            ema_params,
            perceptual_params,
            images,
            global_step
    ):
        def loss_fn(dcae_params, disc_params):
            reconstructed_images = dcae_apply_fn(dcae_params, images)

            reconstruction_loss = jnp.mean(
                jnp.abs(
                    reconstructed_images.astype(jnp.float32) -
                    images.astype(jnp.float32)
                )
            )
            reconstruction_loss *= reconstruction_weight

            perceptual_loss = perceptual_apply_fn(
                perceptual_params,
                reconstructed_images.astype(jnp.float32),
                images.astype(jnp.float32)
            )

            discriminator_factor = adopt_weight(global_step, disc_start)
            d_weight = discriminator_weight

            logits_real = disc_apply_fn(disc_params, images).astype(jnp.float32)
            logits_fake = disc_apply_fn(disc_params, reconstructed_images).astype(jnp.float32)

            generator_loss = -jnp.mean(logits_fake)

            disc_loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
            disc_loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))

            generator_loss = (
                    reconstruction_loss
                    + perceptual_weight * perceptual_loss
                    + generator_weight * discriminator_factor * generator_loss
            )

            disc_loss = (
                discriminator_factor * 0.5 * (disc_loss_real + disc_loss_fake)
            )

            loss_dict = dict(
                generator_loss=generator_loss,
                reconstructed_images=reconstructed_images.astype(jnp.float32),
                reconstruction_loss=reconstruction_loss,
                perceptual_loss=(perceptual_weight * perceptual_loss),
                weighted_gan_loss=(d_weight * discriminator_factor * generator_loss),
                discriminator_factor=discriminator_factor,
                disc_loss_real=disc_loss_real,
                disc_loss_fake=disc_loss_fake,
                disc_loss=disc_loss,
            )

            return (generator_loss, disc_loss), (loss_dict)

        (g_loss, d_loss), func_vjp, (loss_dict) = jax.vjp(
            loss_fn, dcae_params, disc_params, has_aux=True)

        grad_g, _ = func_vjp(jnp.array(1., dtype=jnp.float32), jnp.array(0., dtype=jnp.float32))
        _, grad_d = func_vjp(jnp.array(0., dtype=jnp.float32), jnp.array(1., dtype=jnp.float32))

        grad_g = jax.lax.pmean(grad_g, axis_name="batch")
        grad_d = jax.lax.pmean(grad_d, axis_name="batch")

        g_updates, dcae_opt_state = dcae_optimizer.update(grad_g, dcae_opt_state, dcae_params)
        new_g_params = optax.apply_updates(dcae_params, g_updates)

        d_updates, disc_opt_state = disc_optimizer.update(grad_d, disc_opt_state, disc_params)
        new_d_params = optax.apply_updates(disc_params, d_updates)

        new_ema_params = ema_update(ema_params, new_g_params, ema_decay)

        return new_g_params, dcae_opt_state, new_d_params, disc_opt_state, new_ema_params, loss_dict

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0, 1, 2, 3))


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dcae_config = config['model']
    disc_config = config['discriminator']
    perceptual_config = config['perceptual']
    dataset_config = config['dataset_params']
    wandb_config = config['wandb']

    seed = dcae_config["seed"]

    transform = transforms.Compose([
        transforms.Resize(dataset_config["img_size"]),
        transforms.CenterCrop(dataset_config["img_size"]),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
    ])

    train_dataset = ImageFolder(
        root=dataset_config['data_path'],
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        num_workers=dataset_config['num_workers'],
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=dataset_config["prefetch_factor"],
    )

    dcae = AutoencoderDC(**dcae_config['params'])
    disc = Discriminator(**disc_config['params'])
    perceptual = PerceptualLoss()

    dcae_optimizer = create_optimizer(
        learning_rate=dcae_config["optim_params"]["learning_rate"],
        weight_decay=dcae_config["optim_params"]["weight_decay"],
        beta1=dcae_config["optim_params"]["beta1"],
        beta2=dcae_config["optim_params"]["beta2"],
    )

    disc_optimizer = create_optimizer(
        learning_rate=disc_config["optim_params"]["learning_rate"],
        weight_decay=disc_config["optim_params"]["weight_decay"],
        beta1=disc_config["optim_params"]["beta1"],
        beta2=disc_config["optim_params"]["beta2"],
    )

    run = wandb.init(
        project=wandb_config['project'],
        name=wandb_config['name'],
        reinit=True,
        config=config
    )
    checkpoint_path = dcae_config['checkpoint_path']

    dummy_input, _ = next(iter(train_loader))

    key = jax.random.PRNGKey(seed)
    key, init_key, disc_key, per_key = jax.random.split(key, 4)

    dcae_params = dcae.init(init_key, np.array(dummy_input))
    disc_params = disc.init(disc_key, np.array(dummy_input))
    perceptual_params = perceptual.init(per_key, jnp.ones((1, 256, 256, 3)), jnp.ones((1, 256, 256, 3)))

    dcae_opt_state = dcae_optimizer.init(dcae_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    ema_params = dcae_params

    generator_update_fn = make_generator_update_fn(
        dcae_apply_fn=dcae.apply,
        dcae_optimizer=dcae_optimizer,
        disc_apply_fn=disc.apply,
        disc_optimizer=disc_optimizer,
        perceptual_apply_fn=perceptual.apply,
        reconstruction_weight=dcae_config['reconstruction_weight'],
        perceptual_weight=perceptual_config['perceptual_weight'],
        discriminator_weight=disc_config['discriminator_weight'],
        generator_weight=dcae_config['generator_weight'],
        disc_start=disc_config['disc_start'],
        ema_decay=dcae_config['ema_decay'],
    )

    replicate = lambda tree: jax.device_put_replicated(tree, jax.local_devices())
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    dcae_params_repl = replicate(dcae_params)
    dcae_opt_state_repl = replicate(dcae_opt_state)
    disc_params_repl = replicate(disc_params)
    disc_opt_state_repl = replicate(disc_opt_state)
    perceptual_params_repl = replicate(perceptual_params)
    ema_params_repl = replicate(ema_params)

    state_template = {
        "params": unreplicate(dcae_params_repl),
        "opt_state": unreplicate(dcae_opt_state_repl),
        "ema_params": unreplicate(ema_params_repl),
        'disc_params': unreplicate(disc_params_repl),
        'disc_opt_state': unreplicate(disc_opt_state_repl),
        "epoch": 0,
        "global_step": 0
    }

    del dcae_params, dcae_opt_state, disc_params, disc_opt_state, ema_params

    loaded_state = load_checkpoint(checkpoint_path, state_template)
    start_epoch = 0
    global_step = 0

    if loaded_state:
        dcae_params_repl = replicate(loaded_state['params'])
        dcae_opt_state_repl = replicate(loaded_state['opt_state'])
        ema_params_repl = replicate(loaded_state['ema_params'])
        disc_params_repl = replicate(loaded_state['disc_params'])
        disc_opt_state_repl = replicate(loaded_state['disc_opt_state'])
        start_epoch = loaded_state['epoch'] + 1
        global_step = loaded_state.get('global_step', 0)

    epochs = dcae_config['epochs']

    def shard(x):
        n, *s = x.shape
        return np.reshape(x, (jax.local_device_count(), n // jax.local_device_count(), *s))

    def unshard(x):
        ndev, bs, *s = x.shape
        return jnp.reshape(x, (ndev * bs, *s))

    global_step_repl = replicate(global_step)

    for epoch in range(start_epoch, epochs):
        for step, images in enumerate(train_loader):
            images = shard(np.array(images))

            (
                dcae_params_repl,
                dcae_opt_state_repl,
                disc_params_repl,
                disc_opt_state_repl,
                ema_params_repl,
                loss_dict,
            ) = generator_update_fn(
                dcae_params_repl,
                dcae_opt_state_repl,
                disc_params_repl,
                disc_opt_state_repl,
                ema_params_repl,
                perceptual_params_repl,
                images,
                global_step_repl
            )

            global_step = int(unreplicate(global_step_repl))

            if global_step % 1000 == 0:
                import io
                import matplotlib.pyplot as plt

                real_img = unshard(images)
                recon_img = unshard(loss_dict['reconstructed_images'])

                def process_vis(img):
                    mean = jnp.expand_dims(jnp.array([0.485, 0.456, 0.406]), axis=(0, 1, 2))
                    std = jnp.expand_dims(jnp.array([0.229, 0.224, 0.225]), axis=(0, 1, 2))
                    img = img * std + mean
                    img = np.clip(img, 0.0, 1.0)
                    return (img * 255).astype(np.uint8)

                real_vis = process_vis(real_img)
                recon_vis = process_vis(recon_img)

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(real_vis)
                axs[0].set_title("Original")
                axs[0].axis("off")
                axs[1].imshow(recon_vis)
                axs[1].set_title("Reconstruction")
                axs[1].axis("off")

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close(fig)

                run.log({"reconstruction": wandb.Image(Image.open(buf))}, step=global_step)

            global_step_repl = global_step_repl + 1

            run.log(
                {
                    "reconstruction_loss": unreplicate(loss_dict['reconstruction_loss']),
                    "perceptual_loss": unreplicate(loss_dict['perceptual_loss']),
                    "weighted_gan_loss": unreplicate(loss_dict['weighted_gan_loss']),
                    "generator_loss": unreplicate(loss_dict['generator_loss']),
                    "discriminator_loss": unreplicate(loss_dict['disc_loss']),
                }
            )

        save_checkpoint(checkpoint_path, {
            "params": unreplicate(dcae_params_repl),
            "opt_state": unreplicate(dcae_opt_state_repl),
            "ema_params": unreplicate(ema_params_repl),
            'disc_params': unreplicate(disc_params_repl),
            'disc_opt_state': unreplicate(disc_opt_state_repl),
            "epoch": epoch,
            "global_step": unreplicate(global_step),
        })


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python train.py config.yaml")
    else:
        main(sys.argv[1])
