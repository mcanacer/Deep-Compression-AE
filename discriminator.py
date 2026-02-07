from typing import Sequence

import jax
import jax.numpy as jnp

import flax.linen as nn


class AdaptiveMaxPool2D(nn.Module):
    output_size: tuple[int, int]
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        n, h, w, c = x.shape
        out_h, out_w = self.output_size

        kernel_h = h // out_h
        kernel_w = w // out_w
        stride_h = kernel_h
        stride_w = kernel_w

        x_fp32 = x.astype(jnp.float32)

        pooled = jax.lax.reduce_window(
            operand=x_fp32,
            init_value=jnp.finfo(jnp.float32).min,
            computation=jax.lax.max,
            window_dimensions=(1, kernel_h, kernel_w, 1),
            window_strides=(1, stride_h, stride_w, 1),
            padding="VALID"
        )

        return pooled.astype(x.dtype)


class BlurBlock(nn.Module):
    kernel: Sequence[int] = (1, 3, 3, 1)

    @nn.compact
    def __call__(self, x):
        kernel_1d = jnp.array(self.kernel, dtype=jnp.float32)
        kernel_2d = kernel_1d[None, :] * kernel_1d[:, None]
        kernel_2d /= jnp.sum(kernel_2d)

        c = x.shape[-1]

        kernel_2d = kernel_2d[:, :, None, None]
        kernel_2d = jnp.tile(kernel_2d, (1, 1, 1, c))

        kernel_2d = kernel_2d.astype(x.dtype)

        out = jax.lax.conv_general_dilated(
            x,
            kernel_2d,
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=c
        )
        return out


class DiscriminatorBlock(nn.Module):
    filters: int
    kernel: Sequence[int]
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.filters, kernel_size=(3, 3), dtype=self.dtype, param_dtype=jnp.float32)(x)
        x = BlurBlock(self.kernel)(x)
        x = nn.GroupNorm(num_groups=32)(x.astype(jnp.float32)).astype(self.dtype)
        x = jax.nn.leaky_relu(x, negative_slope=0.1)
        return x


class Discriminator(nn.Module):
    filters: int
    channel_multipliers: Sequence[int]
    blur_kernel_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.filters, kernel_size=(5, 5), dtype=self.dtype, param_dtype=jnp.float32)(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.1)

        BLUR_KERNEL_MAP = {
            3: (1, 2, 1),
            4: (1, 3, 3, 1),
            5: (1, 4, 6, 4, 1),
        }

        filters = self.filters
        num_blocks = len(self.channel_multipliers)
        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            x = DiscriminatorBlock(filters, BLUR_KERNEL_MAP[self.blur_kernel_size], dtype=self.dtype)(x)

        x = AdaptiveMaxPool2D((16, 16))(x)

        x = nn.Conv(filters, kernel_size=(1, 1), dtype=self.dtype, param_dtype=jnp.float32)(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.1)
        x = nn.Conv(1, kernel_size=(5, 5), dtype=self.dtype, param_dtype=jnp.float32)(x)

        return x
