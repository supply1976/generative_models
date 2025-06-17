import tensorflow as tf
from tensorflow import keras


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class TimeEmbedding(keras.layers.Layer):
    """Sinusoidal time embedding layer."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("`dim` must be a positive even integer.")
        self.dim = dim

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        if len(inputs.shape) != 1:
            raise ValueError("Input tensor must be 1D.")
        emb = tf.exp(tf.linspace(0.0, 1.0, self.dim // 2) * -tf.math.log(10000.0))
        emb = inputs[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def TimeMLP(units, actf=keras.activations.swish):
    def apply(inputs):
        temb = keras.layers.Dense(units, activation=actf, kernel_initializer=kernel_init(1.0))(inputs)
        temb = keras.layers.Dense(units, activation=None, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply


class SpaceToDepthLayer(keras.layers.Layer):
    def __init__(self, block_size, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.space_to_depth(inputs, self.block_size)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DepthToSpaceLayer(keras.layers.Layer):
    def __init__(self, block_size, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.block_size)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ResidualBlock(width, attention, num_heads, groups, actf):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv2D(filters=width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

        temb = keras.layers.Activation(actf)(t)
        temb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)
        temb = keras.layers.Reshape([1, 1, -1])(temb)

        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = keras.layers.Activation(actf)(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = keras.layers.Add()([x, temb])
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = keras.layers.Activation(actf)(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = keras.layers.Add()([x, residual])
        if attention:
            res_output = x
            x = keras.layers.GroupNormalization(groups=groups)(x)
            x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=width, attention_axes=(1, 2))(x, x)
            x = keras.layers.Add()([x, res_output])
        return x
    return apply


def DownSample(width):
    def apply(x):
        x = keras.layers.Conv2D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = keras.layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply
