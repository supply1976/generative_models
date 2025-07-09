import tensorflow as tf
from tensorflow import keras


def kernel_init(scale):
    """
    Returns a Keras VarianceScaling initializer with the given scale.
    Args:
        scale (float): Scaling factor for the initializer. Must be positive.
    Returns:
        keras.initializers.VarianceScaling: The initializer instance.
    Raises:
        ValueError: If scale is not a number or is negative.
    """
    if not isinstance(scale, (int, float)):
        raise ValueError("scale must be a number.")
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class TimeEmbedding(keras.layers.Layer):
    """
    Sinusoidal time embedding layer.
    Args:
        dim (int): The embedding dimension. Must be a positive even integer.
    Input shape:
        1D tensor of shape (batch,)
    Output shape:
        2D tensor of shape (batch, dim)
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(dim, int) or dim <= 0 or dim % 2 != 0:
            raise ValueError("`dim` must be a positive even integer.")
        self.dim = dim

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        if len(inputs.shape) != 1:
            raise ValueError("Input tensor must be 1D (batch,). Got shape: {}".format(inputs.shape))
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
    """
    Returns a function that applies two Dense layers to the input, with optional activation.
    Args:
        units (int): Number of units for Dense layers.
        actf (callable): Activation function for the first Dense layer.
    Returns:
        function: A function that applies the MLP to an input tensor.
    """
    def apply(inputs):
        if not hasattr(inputs, 'shape'):
            raise ValueError("Input must be a tensor.")
        temb = keras.layers.Dense(units, activation=actf, kernel_initializer=kernel_init(1.0))(inputs)
        temb = keras.layers.Dense(units, activation=None, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply


# This is a custom Keras Layer version of ResidualBlock.
# It allows for more flexibility in using it as a standalone layer in Keras models.
# It can be used in a Keras Sequential model or Functional API.
# It also supports serialization and deserialization with get_config() and from_config().
# It is similar to the original ResidualBlock function but encapsulated in a Keras Layer
# for better integration with Keras workflows.
class ResidualBlockLayer(keras.layers.Layer):
    """
    Custom Keras Layer version of ResidualBlock.
    Args:
        width (int): Number of output channels.
        attention (bool): Whether to use attention.
        num_heads (int): Number of attention heads.
        groups (int): Number of groups for GroupNormalization.
        actf (callable): Activation function.
        dropout_rate (float): Dropout rate.
        kernel_size (int): Convolution kernel size.
        use_cross_attention (bool): Whether to use cross-attention.
    Input: (x, t)
        x: 4D tensor (batch, height, width, channels)
        t: time embedding tensor
    Output:
        Output tensor after residual block and optional attention.
    """
    def __init__(self, width, attention, num_heads, groups, actf,
                 dropout_rate=0.0, kernel_size=3, use_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.attention = attention
        self.num_heads = num_heads
        self.groups = groups
        self.actf = actf
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.use_cross_attention = use_cross_attention
        # Layers will be built in build()

    def build(self, input_shape):
        # input_shape: [(batch, h, w, c), (batch, ...)]
        input_width = input_shape[0][-1]
        self.res_conv = None
        if input_width != self.width:
            self.res_conv = keras.layers.Conv2D(filters=self.width, kernel_size=1, kernel_initializer=kernel_init(1.0))
        self.temb_act = keras.layers.Activation(self.actf)
        self.temb_dense = keras.layers.Dense(self.width, kernel_initializer=kernel_init(1.0))
        self.temb_reshape = keras.layers.Reshape([1, 1, self.width])
        self.norm1 = keras.layers.GroupNormalization(groups=self.groups)
        self.act1 = keras.layers.Activation(self.actf)
        self.conv1 = keras.layers.Conv2D(self.width, kernel_size=self.kernel_size, padding="same", kernel_initializer=kernel_init(1.0))
        self.dropout = keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.norm2 = keras.layers.GroupNormalization(groups=self.groups)
        self.act2 = keras.layers.Activation(self.actf)
        self.conv2 = keras.layers.Conv2D(self.width, kernel_size=self.kernel_size, padding="same", kernel_initializer=kernel_init(0.0))
        if self.attention:
            self.norm_attn = keras.layers.GroupNormalization(groups=self.groups)
            self.mha = keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.width, attention_axes=(1, 2))
        super().build(input_shape)

    def call(self, inputs):
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError("inputs must be a tuple/list of (x, t)")
        x, t = inputs
        if x.shape.rank != 4:
            raise ValueError("x must be a 4D tensor (batch, height, width, channels)")
        residual = x if self.res_conv is None else self.res_conv(x)
        temb = self.temb_act(t)
        temb = self.temb_dense(temb)
        temb = self.temb_reshape(temb)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = keras.layers.Add()([x, temb])
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = keras.layers.Add()([x, residual])
        if self.attention:
            res_output = x
            x = self.norm_attn(x)
            if self.use_cross_attention:
                ctx = tf.broadcast_to(temb, tf.shape(x))
                x = self.mha(x, ctx)
            else:
                x = self.mha(x, x)
            x = keras.layers.Add()([x, res_output])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "width": self.width,
            "attention": self.attention,
            "num_heads": self.num_heads,
            "groups": self.groups,
            "actf": self.actf,
            "dropout_rate": self.dropout_rate,
            "kernel_size": self.kernel_size,
            "use_cross_attention": self.use_cross_attention
        })
        return config
    
    def from_config(cls, config):
        return cls(**config)


# Use function returning apply
# This is a function that returns another function (apply) which applies the residual block.
def ResidualBlock(width, attention, num_heads, groups, actf,
                  dropout_rate=0.0, kernel_size=3, use_cross_attention=False):
    """
    Returns a function that applies a residual block with optional attention and cross-attention.
    Args:
        width (int): Number of output channels.
        attention (bool): Whether to use attention.
        num_heads (int): Number of attention heads.
        groups (int): Number of groups for GroupNormalization.
        actf (callable): Activation function.
        dropout_rate (float): Dropout rate.
        kernel_size (int): Convolution kernel size.
        use_cross_attention (bool): Whether to use cross-attention.
    Returns:
        function: A function that applies the residual block to (x, t).
        
    diagram:
    (x, t)  # x: feature map, t: time embedding
    |
    |-----------------------------.
    |                             |
    |                         [If input_width != width]
    |                             |
    |                         1x1 Conv2D
    |                             |
    |------------------------> residual
    |
    |-- GroupNorm --> Activation --> Conv2D (kernel_size)
    |                             |
    |                        [Process t:]
    |                        Activation --> Dense --> Reshape (1,1,-1)
    |                             |
    |------------------------> Add (x + temb)
    |
    |-- [Dropout if needed]
    |-- GroupNorm --> Activation --> Conv2D (kernel_size)
    |                             |
    |------------------------> Add (x + residual)
    |
    |-- [If attention:]
    |     |-- GroupNorm
    |     |-- [If cross-attention:]
    |     |     Broadcast temb to x shape
    |     |     MultiHeadAttention(x, temb)
    |     |-- [Else:]
    |     |     MultiHeadAttention(x, x)
    |     |-- Add (attention + res_output)
    |
    v
    output
    """
    def apply(inputs):
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError("inputs must be a tuple/list of (x, t)")
        x, t = inputs
        if x.shape.rank != 4:
            raise ValueError("x must be a 4D tensor (batch, height, width, channels)")
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
        x = keras.layers.Conv2D(width, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = keras.layers.Add()([x, temb])
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = keras.layers.Activation(actf)(x)
        x = keras.layers.Conv2D(width, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = keras.layers.Add()([x, residual])
        if attention:
            res_output = x
            x = keras.layers.GroupNormalization(groups=groups)(x)
            if use_cross_attention:
                ctx = tf.broadcast_to(temb, tf.shape(x))
                x = keras.layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=width, attention_axes=(1, 2)
                )(x, ctx)
            else:
                x = keras.layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=width, attention_axes=(1, 2)
                )(x, x)
            x = keras.layers.Add()([x, res_output])
        return x
    return apply


def DownSample(width):
    """
    Returns a function that applies a strided convolution to downsample the input.
    Args:
        width (int): Number of output channels.
    Returns:
        function: A function that downsamples a 4D tensor.
    """
    def apply(x):
        if x.shape.rank != 4:
            raise ValueError("Input to DownSample must be a 4D tensor (batch, height, width, channels)")
        x = keras.layers.Conv2D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    """
    Returns a function that upsamples the input and applies a convolution.
    Args:
        width (int): Number of output channels.
        interpolation (str): Interpolation method for upsampling.
    Returns:
        function: A function that upsamples a 4D tensor.
    """
    def apply(x):
        if x.shape.rank != 4:
            raise ValueError("Input to UpSample must be a 4D tensor (batch, height, width, channels)")
        x = keras.layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply
