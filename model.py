import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Optional, Tuple, Any, Union

import flax.linen as nn


def get_activation(act_fn):
    if act_fn == 'relu':
        return jax.nn.relu
    elif act_fn == 'leakyrelu':
        return jax.nn.leaky_relu
    elif act_fn == 'silu':
        return jax.nn.silu


class RMSNorm(nn.Module):
    epsilon: float = 1e-5
    use_scale: bool = True
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)

        if self.use_scale:
            scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
            x = x * scale

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],))
            x = x + bias

        return x


class DCDownBlock2d(nn.Module):
    out_channels: int
    factor: int =  2
    shortcut: bool = False

    @nn.compact
    def __call__(self, hidden_states):
        out_ratio = self.factor ** 2
        out_channels = self.out_channels // out_ratio
        group_size = hidden_states.shape[-1] * out_ratio // self.out_channels
        x = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
        )(hidden_states)

        # Pixel Unshuffle
        x = rearrange(x, 'n (h b1) (w b2) c -> n h w (c b1 b2)', b1=self.factor, b2=self.factor)

        if self.shortcut:
            y = rearrange(hidden_states, 'n (h b1) (w b2) c -> n h w (c b1 b2)', b1=self.factor, b2=self.factor)
            y = y.reshape(*y.shape[:-1], -1, group_size)
            y = y.mean(axis=-1)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):
    out_channels: int
    shortcut: bool = True
    factor: int = 2

    @nn.compact
    def __call__(self, hidden_states):
        repeats = self.out_channels * self.factor ** 2 // hidden_states.shape[-1]
        out_ratio = self.factor ** 2
        conv_out_channels = self.out_channels * out_ratio

        x = nn.Conv(conv_out_channels, kernel_size=(3, 3))(hidden_states)

        # Pixel shuffle
        x = rearrange(x, 'n h w (c b1 b2) -> n (h b1) (w b2) c',b1=self.factor, b2=self.factor)

        if self.shortcut:
            y = jnp.repeat(hidden_states, repeats, axis=-1)

            # Pixel shuffle
            y = rearrange(y, 'n h w (c b1 b2) -> n (h b1) (w b2) c', b1=self.factor, b2=self.factor)

            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class SanaMultiscaleAttentionProjection(nn.Module):
    num_attention_heads: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=(self.kernel_size, self.kernel_size),
            feature_group_count=x.shape[-1],
            use_bias=False,
            padding="SAME"
        )(x)

        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=(1, 1),
            feature_group_count=3 * self.num_attention_heads,
            use_bias=False,
        )(x)

        return x


class SanaMultiscaleLinearAttention(nn.Module):
    in_channels: int
    out_channels: int
    num_attention_heads: Optional[int] = None
    attention_head_dim: int = 8
    mult: float = 1.0
    norm_type: str = "rms"
    kernel_sizes: Tuple[int, ...] = (5,)
    eps: float = 1e-15
    residual_connection: bool = False

    def apply_linear_attention(self, query, key, value):
        padding = jnp.ones((value.shape[0], value.shape[1], 1, value.shape[3]), dtype=value.dtype)
        value_padded = jnp.concatenate([value, padding], axis=2)

        context = jnp.einsum('bhdl,bhkl->bhdk', value_padded, key)

        hidden_states = jnp.einsum('bhdk,bhkl->bhdl', context, query)

        hidden_states = hidden_states.astype(jnp.float32)
        numerator = hidden_states[:, :, :-1, :]
        denominator = hidden_states[:, :, -1:, :] + self.eps

        return (numerator / denominator).astype(query.dtype)

    def apply_quadratic_attention(self, query, key, value):
        scores = jnp.einsum('bhdm,bhdn->bhmn', key, query)
        scores = scores.astype(jnp.float32)
        scores = scores / (jnp.sum(scores, axis=2, keepdims=True) + self.eps)

        hidden_states = jnp.einsum('bhdn,bhmn->bhdm', value, scores.astype(query.dtype))
        return hidden_states

    @nn.compact
    def __call__(self, hidden_states):
        N, H, W, C = hidden_states.shape
        L = H * W

        use_linear_attention = (L > self.attention_head_dim)
        residual = hidden_states if self.residual_connection else None

        n_heads = self.num_attention_heads or int(self.in_channels // self.attention_head_dim * self.mult)

        num_scales = 1 + len(self.kernel_sizes)
        eff_head_dim = self.attention_head_dim * num_scales
        inner_dim = n_heads * self.attention_head_dim

        query = nn.Dense(inner_dim, use_bias=False, name="to_q")(hidden_states)
        key = nn.Dense(inner_dim, use_bias=False, name="to_k")(hidden_states)
        value = nn.Dense(inner_dim, use_bias=False, name="to_v")(hidden_states)

        qkv = jnp.concatenate([query, key, value], axis=-1)

        multi_scale_qkv = [qkv]
        for i, k_size in enumerate(self.kernel_sizes):
            proj = SanaMultiscaleAttentionProjection(
                num_attention_heads=n_heads,
                kernel_size=k_size,
                name=f"ms_proj_{i}"
            )(qkv)
            multi_scale_qkv.append(proj)

        x = jnp.concatenate(multi_scale_qkv, axis=-1)

        x = x.reshape(N, L, n_heads, 3, eff_head_dim)
        x = jnp.transpose(x, (0, 2, 4, 3, 1))

        query, key, value = jnp.split(x, 3, axis=3)
        query, key, value = query[:, :, :, 0, :], key[:, :, :, 0, :], value[:, :, :, 0, :]

        query = jax.nn.relu(query)
        key = jax.nn.relu(key)

        if use_linear_attention:
            attn_out = self.apply_linear_attention(query, key, value)
        else:
            attn_out = self.apply_quadratic_attention(query, key, value)

        attn_out = jnp.transpose(attn_out, (0, 3, 1, 2))
        attn_out = attn_out.reshape(N, H, W, -1)

        x = nn.Dense(self.out_channels, use_bias=False, name="to_out")(attn_out)

        if self.norm_type == "rms":
            x = RMSNorm()(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)

        if residual is not None:
            x = x + residual

        return x


class GLUMBConv(nn.Module):
    out_channels: int
    norm_type: str = "rms"
    expansion: int = 4
    residual_connection: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x if self.residual_connection else None
        hidden_channels = int(x.shape[-1] * self.expansion)

        x = nn.Conv(hidden_channels * 2, kernel_size=(1, 1))(x)
        x = get_activation('silu')(x)

        x = nn.Conv(
            hidden_channels * 2,
            kernel_size=(3, 3),
            feature_group_count=hidden_channels * 2
        )(x)

        hidden_states, gate = jnp.split(x, 2, axis=-1)
        hidden_states = hidden_states * get_activation('silu')(gate)

        x = nn.Conv(self.out_channels, kernel_size=(1, 1), use_bias=False)(hidden_states)

        if self.norm_type == "rms":
            x = RMSNorm()(x)

        x = x + residual if self.residual_connection else x
        return x


class ResBlock(nn.Module):
    out_channels: int
    act_fn: str
    norm_type: str = 'rms'

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(x.shape[-1], kernel_size=(3, 3))(x)
        x = get_activation(self.act_fn)(x)
        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            use_bias=False,
        )(x)

        if self.norm_type == 'rms':
            x = RMSNorm()(x)

        return x + residual


class Encoder(nn.Module):
    in_channels: int
    latent_channels: int
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    layers_per_block: Tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    attention_head_dim: int = 32
    qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,))
    norm_type: str = "rms"
    act_fn: str = "silu"
    out_shortcut: bool = True

    @nn.compact
    def __call__(self, x):
        num_blocks = len(self.block_out_channels)
        b_types = (self.block_type,) * num_blocks if isinstance(self.block_type, str) else self.block_type

        if self.layers_per_block[0] > 0:
            x = nn.Conv(features=self.block_out_channels[0], kernel_size=(3, 3))(x)
        else:
            x = DCDownBlock2d(
                out_channels=self.block_out_channels[0] if self.layers_per_block[0] > 0 else self.block_out_channels[1],
                shortcut=False
            )(x)

        for i, (out_ch, num_layers) in enumerate(zip(self.block_out_channels, self.layers_per_block)):
            for j in range(num_layers):
                if b_types[i] == "ResBlock":
                    x = ResBlock(out_channels=out_ch, act_fn=self.act_fn, norm_type=self.norm_type)(x)
                elif b_types[i] == "EfficientViTBlock":
                    x = SanaMultiscaleLinearAttention(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        attention_head_dim=self.attention_head_dim,
                        kernel_sizes=self.qkv_multiscales[i],
                        norm_type=self.norm_type,
                        residual_connection=True
                    )(x)
                    x = GLUMBConv(
                        out_channels=out_ch,
                        norm_type=self.norm_type,
                        residual_connection=True
                    )(x)

            if i < num_blocks - 1 and num_layers > 0:
                x = DCDownBlock2d(out_channels=self.block_out_channels[i + 1], shortcut=True)(x)

        if self.out_shortcut:
            group_size = x.shape[-1] // self.latent_channels
            shortcut = x.reshape((*x.shape[:-1], self.latent_channels, group_size)).mean(axis=-1)
            x = nn.Conv(features=self.latent_channels, kernel_size=(3, 3))(x) + shortcut
        else:
            x = nn.Conv(features=self.latent_channels, kernel_size=(3, 3))(x)

        return x


class Decoder(nn.Module):
    in_channels: int
    latent_channels: int
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    layers_per_block: Tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    attention_head_dim: int = 32
    qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,))
    norm_type: Union[str, Tuple[str]] = "rms",
    act_fn: Union[str, Tuple[str]] = "silu",
    in_shortcut: bool = True
    conv_act_fn: str = "relu"

    @nn.compact
    def __call__(self, z):
        num_blocks = len(self.block_out_channels)
        b_types = (self.block_type,) * num_blocks if isinstance(self.block_type, str) else self.block_type
        norm_types = (self.norm_type,) * num_blocks if isinstance(self.norm_type, str) else self.norm_type
        act_fn = (self.act_fn,) * num_blocks if isinstance(self.act_fn, str) else self.act_fn

        if self.in_shortcut:
            repeats = self.block_out_channels[-1] // self.latent_channels
            shortcut = jnp.repeat(z, repeats, axis=-1)
            x = nn.Conv(features=self.block_out_channels[-1], kernel_size=(3, 3))(z) + shortcut
        else:
            x = nn.Conv(features=self.block_out_channels[-1], kernel_size=(3, 3))(z)

        for i in range(num_blocks - 1, -1, -1):
            out_ch = self.block_out_channels[i]
            num_layers = self.layers_per_block[i]

            if i < num_blocks - 1 and num_layers > 0:
                x = DCUpBlock2d(out_channels=out_ch, shortcut=True)(x)

            for j in range(num_layers):
                if b_types[i] == "ResBlock":
                    x = ResBlock(out_channels=out_ch, norm_type=norm_types[i], act_fn=act_fn[i])(x)
                elif b_types[i] == "EfficientViTBlock":
                    x = SanaMultiscaleLinearAttention(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        attention_head_dim=self.attention_head_dim,
                        kernel_sizes=self.qkv_multiscales[i],
                        residual_connection=True
                    )(x)
                    x = GLUMBConv(out_channels=out_ch, residual_connection=True)(x)

        x = RMSNorm()(x)
        x = get_activation(self.conv_act_fn)(x)

        if self.layers_per_block[0] > 0:
            x = nn.Conv(features=self.in_channels, kernel_size=(3, 3))(x)
        else:
            x = DCUpBlock2d(out_channels=self.in_channels, shortcut=False)(x)

        return x


class AutoencoderDC(nn.Module):
    in_channels: int = 3
    latent_channels: int = 32
    attention_head_dim: int = 32
    encoder_block_types: Any = "ResBlock"
    decoder_block_types: Any = "ResBlock"
    encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 3, 3, 3)
    decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3, 3, 3)
    encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,))
    decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,))
    decoder_norm_types: Union[str, Tuple[str]] = "rms",
    decoder_act_fns: Union[str, Tuple[str]] = "silu",
    encoder_out_shortcut: bool = True
    decoder_in_shortcut: bool = True
    decoder_conv_act_fn: str = "relu"
    scaling_factor: float = 1.0

    @nn.compact
    def __call__(self, x):
        z = Encoder(
            in_channels=self.in_channels,
            latent_channels=self.latent_channels,
            block_out_channels=self.encoder_block_out_channels,
            layers_per_block=self.encoder_layers_per_block,
            block_type=self.encoder_block_types,
            attention_head_dim=self.attention_head_dim,
            qkv_multiscales=self.encoder_qkv_multiscales
        )(x)

        recon = Decoder(
            in_channels=self.in_channels,
            latent_channels=self.latent_channels,
            block_out_channels=self.decoder_block_out_channels,
            layers_per_block=self.decoder_layers_per_block,
            block_type=self.decoder_block_types,
            attention_head_dim=self.attention_head_dim,
            norm_type=self.decoder_norm_types,
            act_fn=self.decoder_act_fns,
            in_shortcut=self.decoder_in_shortcut,
            qkv_multiscales=self.decoder_qkv_multiscales,
            conv_act_fn=self.decoder_conv_act_fn,
        )(z)

        return recon

    def encode(self, x):
        z = Encoder(
            in_channels=self.in_channels,
            latent_channels=self.latent_channels,
            block_out_channels=self.encoder_block_out_channels,
            layers_per_block=self.encoder_layers_per_block,
            block_type=self.encoder_block_types,
            attention_head_dim=self.attention_head_dim,
            qkv_multiscales=self.encoder_qkv_multiscales
        )(x)
        return z

    def decode(self, z):
        z = z * (1.0 / self.scaling_factor)
        return Decoder(
            in_channels=self.in_channels,
            latent_channels=self.latent_channels,
            block_out_channels=self.decoder_block_out_channels,
            layers_per_block=self.decoder_layers_per_block,
            block_type=self.decoder_block_types,
            attention_head_dim=self.attention_head_dim,
            qkv_multiscales=self.decoder_qkv_multiscales
        )(z)


if __name__ == "__main__":
    config = {
        "in_channels": 3,
        "latent_channels": 32,
        "encoder_block_types": ("ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock", "EfficientViTBlock",
                                "EfficientViTBlock"),
        "decoder_block_types": ("ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock", "EfficientViTBlock",
                                "EfficientViTBlock"),
        "encoder_block_out_channels": (128, 256, 512, 512, 1024, 1024),
        "decoder_block_out_channels": (128, 256, 512, 512, 1024, 1024),
        "encoder_layers_per_block": (0, 4, 8, 2, 2, 2),
        "decoder_layers_per_block": (0, 5, 10, 2, 2, 2),
        "encoder_qkv_multiscales": ((), (), (), (), (), ()),
        "decoder_qkv_multiscales": ((), (), (), (), (), ()),
        "decoder_norm_types": ("batch", "batch", "batch", "rms", "rms", "rms"),
        "decoder_act_fns": ("relu", "relu", "relu", "silu", "silu", "silu")
    }

    model = AutoencoderDC(**config)
