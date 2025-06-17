import tensorflow as tf
from tensorflow import keras

from layers import (
    kernel_init,
    TimeEmbedding,
    TimeMLP,
    SpaceToDepthLayer,
    DepthToSpaceLayer,
    ResidualBlock,
    DownSample,
    UpSample,
)


def build_model(
    image_size=256,
    image_channel=1,
    widths=[32, 64, 128, 256],
    has_attention=[False, False, False, True],
    num_heads=1,
    num_res_blocks=2,
    norm_groups=32,
    interpolation="nearest",
    actf=keras.activations.swish,
    block_size=2,
    temb_dim=128,
):
    """Build UNet model."""
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("`image_size` must be a positive integer.")
    if not isinstance(image_channel, int) or image_channel <= 0:
        raise ValueError("`image_channel` must be a positive integer.")
    if not isinstance(widths, list) or len(widths) == 0:
        raise ValueError("`widths` must be a non-empty list of integers.")
    if not all(isinstance(w, int) and w > 0 for w in widths):
        raise ValueError("All elements in `widths` must be positive integers.")
    if not isinstance(has_attention, list) or len(has_attention) != len(widths):
        raise ValueError("`has_attention` must be a list of booleans with the same length as `widths`." )
    if not all(isinstance(h, bool) for h in has_attention):
        raise ValueError("All elements in `has_attention` must be booleans.")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("`num_heads` must be a positive integer.")
    if not isinstance(num_res_blocks, int) or num_res_blocks <= 0:
        raise ValueError("`num_res_blocks` must be a positive integer.")
    if not isinstance(norm_groups, int) or norm_groups <= 0:
        raise ValueError("`norm_groups` must be a positive integer.")
    if not isinstance(interpolation, str):
        raise ValueError("`interpolation` must be a string.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("`block_size` must be a positive integer.")
    if not isinstance(temb_dim, int) or temb_dim <= 0:
        raise ValueError("`temb_dim` must be a positive integer.")

    input_shape = (image_size, image_size, image_channel)
    image_input = keras.Input(shape=input_shape, name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int32, name="time_input")

    if block_size > 1:
        assert image_size % block_size == 0
        x = SpaceToDepthLayer(block_size)(image_input)
    else:
        x = image_input

    x = keras.layers.Conv2D(
        filters=widths[0],
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)

    temb = TimeEmbedding(dim=temb_dim, name="TimeEmb")(time_input)
    temb = TimeMLP(units=temb_dim, actf=actf)(temb)

    skips = [x]

    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], has_attention[i], num_heads=num_heads, groups=norm_groups, actf=actf
            )([x, temb])
            skips.append(x)
        if i != len(widths) - 1:
            x = DownSample(widths[i])(x)
            skips.append(x)

    x = ResidualBlock(widths[-1], False, num_heads=num_heads, groups=norm_groups, actf=actf)([x, temb])
    x = ResidualBlock(widths[-1], False, num_heads=num_heads, groups=norm_groups, actf=actf)([x, temb])

    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], has_attention[i], num_heads=num_heads, groups=norm_groups, actf=actf)([x, temb])
        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    x = keras.layers.GroupNormalization(groups=norm_groups)(x)
    x = keras.layers.Activation(actf)(x)
    x = keras.layers.Conv2D(
        image_channel * (block_size ** 2),
        (3, 3),
        padding="same",
        kernel_initializer=kernel_init(0.0),
        name="final_conv2d",
    )(x)

    if block_size > 1:
        x = DepthToSpaceLayer(block_size)(x)
    return keras.Model([image_input, time_input], x, name="unet")

