"""
unet.py
--------
Defines a configurable UNet model for image-to-image tasks with optional attention and time embedding support.

Functions:
    build_model(...):
        Builds and returns a Keras UNet model with skip connections, residual blocks, and optional attention.

Example usage:
    from unet import build_model
    model = build_model(image_size=256, image_channel=1)
"""

import tensorflow as tf
from tensorflow import keras

from layers import (
    kernel_init,
    TimeEmbedding,
    TimeMLP,
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
    dropout_rate=0.0,
    kernel_size=3,
    use_cross_attention=False,
    num_classes=None,
    class_emb_dim=None,
):
    """
    Build a configurable UNet model with skip connections, residual blocks, and optional attention.

    Parameters
    ----------
    image_size : int
        Spatial resolution of the input and output tensors.
    image_channel : int
        Number of channels in the input image.
    widths : list[int]
        Channel widths for each resolution level.
    has_attention : list[bool]
        Whether to apply self-attention at each level. Length must match widths.
    num_heads : int
        Number of attention heads when attention is enabled.
    num_res_blocks : int
        Number of residual blocks at each level.
    norm_groups : int
        Number of groups for GroupNormalization.
    interpolation : str
        Upsampling interpolation method.
    actf : Callable
        Activation function used throughout the network.
    block_size : int
        Space-to-depth scaling factor.
    temb_dim : int
        Dimension of the time embedding.
    dropout_rate : float
        Dropout rate applied inside residual blocks.
    kernel_size : int or tuple[int, int]
        Convolution kernel size used for all convolutions.
    use_cross_attention : bool
        If True, attention blocks use cross-attention with the time embeddings.
    num_classes : int or None
        Number of classes for class conditioning. If provided, a label input is
        added and combined with the time embedding.
    class_emb_dim : int or None
        Dimension of the class embedding. Defaults to ``temb_dim`` when
        ``num_classes`` is specified.

    Returns
    -------
    keras.Model
        Model that maps [image, timestep] inputs to an image tensor of
        shape (batch, image_size, image_size, image_channel).
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("`image_size` must be a positive integer.")
    if not isinstance(image_channel, int) or image_channel <= 0:
        raise ValueError("`image_channel` must be a positive integer.")
    if not isinstance(widths, list) or len(widths) == 0:
        raise ValueError("`widths` must be a non-empty list of integers.")
    if not all(isinstance(w, int) and w > 0 for w in widths):
        raise ValueError("All elements in `widths` must be positive integers.")
    if not isinstance(has_attention, list) or len(has_attention) != len(widths):
        raise ValueError("`has_attention` must be a list of booleans with the same length as `widths`.")
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
    if not isinstance(dropout_rate, (int, float)) or not 0 <= dropout_rate <= 1:
        raise ValueError("`dropout_rate` must be between 0 and 1.")
    if isinstance(kernel_size, int):
        if kernel_size <= 0:
            raise ValueError("`kernel_size` must be positive.")
        kernel_size = (kernel_size, kernel_size)
    elif (
        isinstance(kernel_size, (list, tuple))
        and len(kernel_size) == 2
        and all(isinstance(k, int) and k > 0 for k in kernel_size)
    ):
        kernel_size = tuple(kernel_size)
    else:
        raise ValueError("`kernel_size` must be an int or tuple of two ints.")
    if not isinstance(use_cross_attention, bool):
        raise ValueError("`use_cross_attention` must be a boolean.")

    input_shape = (image_size, image_size, image_channel)
    image_input = keras.Input(shape=input_shape, name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int32, name="time_input")

    # Space-to-depth if block_size > 1
    if block_size > 1:
        assert image_size % block_size == 0
        x = tf.nn.space_to_depth(image_input, block_size)
    else:
        x = image_input

    # Initial convolution
    x = keras.layers.Conv2D(
        filters=widths[0],
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)

    # Time embedding
    temb = TimeEmbedding(dim=temb_dim, name="TimeEmb")(time_input)
    temb = TimeMLP(units=temb_dim, actf=actf)(temb)

    inputs = [image_input, time_input]
    if num_classes is not None:
        if class_emb_dim is None:
            class_emb_dim_ = temb_dim
        else:
            class_emb_dim_ = class_emb_dim
        class_input = keras.Input(shape=(), dtype=tf.int32, name="class_input")
        cemb = keras.layers.Embedding(num_classes, class_emb_dim_)(class_input)
        cemb = TimeMLP(units=temb_dim, actf=actf)(cemb)
        temb = keras.layers.Add()([temb, cemb])
        inputs.append(class_input)
    else:
        class_input = None

    skips = [x]

    # Downsampling path
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i],
                has_attention[i],
                num_heads=num_heads,
                groups=norm_groups,
                actf=actf,
                dropout_rate=dropout_rate,
                kernel_size=kernel_size,
                use_cross_attention=use_cross_attention,
            )([x, temb])
            skips.append(x)
        if i != len(widths) - 1:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # Bottleneck
    x = ResidualBlock(
        widths[-1],
        True,
        num_heads=num_heads,
        groups=norm_groups,
        actf=actf,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        use_cross_attention=use_cross_attention,
    )([x, temb])
    x = ResidualBlock(
        widths[-1],
        False,
        num_heads=num_heads,
        groups=norm_groups,
        actf=actf,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        use_cross_attention=use_cross_attention,
    )([x, temb])

    # Upsampling path
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i],
                has_attention[i],
                num_heads=num_heads,
                groups=norm_groups,
                actf=actf,
                dropout_rate=dropout_rate,
                kernel_size=kernel_size,
                use_cross_attention=use_cross_attention,
            )([x, temb])
        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # Final normalization and convolution
    x = keras.layers.GroupNormalization(groups=norm_groups)(x)
    x = keras.layers.Activation(actf)(x)
    x = keras.layers.Conv2D(
        image_channel * (block_size ** 2),
        kernel_size,
        padding="same",
        kernel_initializer=kernel_init(0.0),
        name="final_conv2d",
    )(x)

    # Depth-to-space if block_size > 1
    if block_size > 1:
        x = tf.nn.depth_to_space(x, block_size)

    return keras.Model(inputs, x, name="unet")

