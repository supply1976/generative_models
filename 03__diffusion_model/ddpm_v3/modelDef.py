import os, sys
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm
import gc

tf.debugging.disable_traceback_filtering()

#tf.config.set_visible_devices([], 'GPU')
#tf.config.set_visible_devices([], 'CPU')
#tf.config.set_logical_device_configuration(
#    tf.config.list_physical_devices('GPU')[0],
#    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
#tf.config.experimental.set_virtual_device_configuration(
#    tf.config.list_physical_devices('GPU')[0],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#tf.config.experimental.set_virtual_device_configuration(
#    tf.config.list_physical_devices('GPU')[1],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#tf.config.experimental.set_virtual_device_configuration(
#    tf.config.list_physical_devices('GPU')[2],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#tf.config.experimental.set_virtual_device_configuration(
#    tf.config.list_physical_devices('GPU')[3],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#tf.config.run_functions_eagerly(True)
#tf.config.optimizer.set_jit(True) # enable XLA
#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
#tf.config.optimizer.set_experimental_options({"layout_optimizer": True})
#tf.config.optimizer.set_experimental_options({"constant_folding": True})
#tf.config.optimizer.set_experimental_options({"shape_optimization": True})
#tf.config.optimizer.set_experimental_options({"remapping": True})
#tf.config.optimizer.set_experimental_options({"arithmetic_optimization": True})
#tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
#tf.config.optimizer.set_experimental_options({"disable_model_pruning": True})


class DiffusionUtility:
  def __init__(self, 
    b0=0.1, b1=20.0, scheduler='linear', timesteps=1000, 
    pred_type='velocity', reverse_stride=1, ddim_eta = 1.0,
    ):
    self.b0 = b0
    self.b1 = b1
    self.scheduler = scheduler
    self.timesteps = timesteps
    # normalize time to 0 ~ 1
    self.timesamps = np.linspace(0, 1, timesteps+1, dtype=np.float64)
    self.eps = 1.0e-6
    self.CLIP_MIN = -1.0
    self.CLIP_MAX = 1.0
    self.reverse_stride = reverse_stride
    self.pred_type = pred_type
    self.ddim_eta = ddim_eta
    assert isinstance(timesteps, int)
    assert isinstance(reverse_stride, int)
    assert reverse_stride >= 1
    assert timesteps % reverse_stride == 0

    alphas, mu_coefs, var_coefs = (None, None, None)
    if self.scheduler == 'linear':
      # same as original DDPM paper, normalize time to 0 ~ 1
      # beta(t) = b0 + (b1-b0)*t, for 0 <= t <= 1
      # integrated_beta(t): B(t) = b0*t + 0.5*(b1-b0)*t^2, for 0 <= t <= 1
      # alpha(t) === exp(-1*B(t))
      # alpha(t, s) === alpha(t)/alpha(s), for t > s >=0
      # for discrete time sampling:
      # alpha(s) = alpha(t-n) for n >= 1, n is the reverse stride
      
      Bt = self.timesamps * self.b0 + 0.5* self.timesamps**2 * (self.b1-self.b0)
      # Bt=[B0,B1,B2,...,BN], B0 = 0
      assert np.all(Bt>=0)
      assert len(Bt)==timesteps+1
      #deltaBt = Bt[reverse_stride:]-Bt[0:-reverse_stride]
      #assert len(deltaBt)==len(Bt)-reverse_stride
      #assert np.all(deltaBt>=0)
      #alpha_ts = np.exp(-1*deltaBt)
      alphas = np.exp(-1*Bt)

    elif self.scheduler == 'cosine':
      # same as iDDPM paper, use cosine scheduler for alpha(t)
      # this directly defines alpha(t) rather than defining beta(t)
      # alpha(t) === exp(-1*B(t)) = cos(t*pi/2)^2  ; for 0 <= t <= 1
      # --> alpha(0) = 1, alpha(1) = 0
      
      # define end_angel in degree, hard code here,
      # do not use exactly 90 degree to avoid numerical issue
      end_angle = 89; # degree
      angles = self.timesamps * end_angle *np.pi / 180  ; # rads
      alphas = np.cos(angles)**2

    elif self.scheduler == 'cos6':
      # New scheduler using cosine to the power of 6
      # found it's better than cosine and linear
      end_angle = 80 ; # degree
      angles = self.timesamps * end_angle *np.pi / 180
      alphas = np.cos(angles)**6
    else:
      print("not supported diffusion scheduler, exit")
      return
    
    assert alphas is not None
    # for forward sampling
    mu_coefs = np.sqrt(alphas)   ; # this is identical to signal_rates in other paper 
    var_coefs = 1 - alphas       ; # this is identical to noise_powers in other paper
    sigma_coefs = np.sqrt(var_coefs) ; # this is noise_rates in other paper
    # define constant Tensors
    self.mu_coefs = tf.constant(mu_coefs, tf.float32)
    self.var_coefs = tf.constant(var_coefs, tf.float32)
    self.sigma_coefs = tf.constant(sigma_coefs, tf.float32)
    
    # for reverse sampling
    # alpha_ts === alpha(t) / alpha(s) for t > s >= 0
    alpha_t = alphas[reverse_stride:]
    alpha_s = alphas[0:-reverse_stride]
    alpha_ts = alpha_t / alpha_s
    var_coefs_st = (1-alpha_s) / (1-alpha_t)
    reverse_var_coefs = var_coefs_st*(1-alpha_ts)
    assert np.all(reverse_var_coefs >= 0)
    reverse_sigma_coefs = np.sqrt(reverse_var_coefs)

    # used in DDPM sampling method
    reverse_mu_ddpm_xt = var_coefs_st * np.sqrt(alpha_ts)
    reverse_mu_ddpm_x0 = np.sqrt(alpha_s) * (1-alpha_ts) / (1-alpha_t)
    # used in DDIM sampling method
    reverse_mu_ddim_x0 = np.sqrt(alpha_s)
    reverse_mu_ddim_noise = np.sqrt(1 - alpha_s - self.ddim_eta*reverse_var_coefs)
    # insert 0 to make the total length = timesteps+1
    reverse_sigma_coefs = np.insert(reverse_sigma_coefs, 0, [0.0]*reverse_stride)
    reverse_mu_ddpm_xt = np.insert(reverse_mu_ddpm_xt, 0, [0.0]*reverse_stride)
    reverse_mu_ddpm_x0 = np.insert(reverse_mu_ddpm_x0, 0, [1.0]*reverse_stride)
    reverse_mu_ddim_x0 = np.insert(reverse_mu_ddim_x0, 0, [1.0]*reverse_stride)
    reverse_mu_ddim_noise = np.insert(reverse_mu_ddim_noise, 0, [0.0]*reverse_stride)
    # define constant Tensors
    self.reverse_sigma_coefs = tf.constant(reverse_sigma_coefs, tf.float32)
    self.reverse_mu_ddpm_xt = tf.constant(reverse_mu_ddpm_xt, tf.float32)
    self.reverse_mu_ddpm_x0 = tf.constant(reverse_mu_ddpm_x0, tf.float32)
    self.reverse_mu_ddim_x0 = tf.constant(reverse_mu_ddim_x0, tf.float32)
    self.reverse_mu_ddim_noise = tf.constant(reverse_mu_ddim_noise, tf.float32)

  def q_sample(self, x_0, t, noise):
    """
      Forward Sampling (noising process)
      x_0: 4D input image tensor, shape=[batch, h, w, c]
      t: 1D integer index tensor, shape=[batch], value=0, 1, 2, 3, ..., N
      noise: 4D gaussian random number tensor, shape=[batch, h, w, c]
    """
    # get coefficients at t using tf.gather
    sigma_t = tf.gather(self.sigma_coefs, t)[:,None,None,None]
    mu_t = tf.gather(self.mu_coefs, t)[:,None,None,None]
    x_t = mu_t * x_0 + sigma_t * noise
    # velocity
    v_t = mu_t * noise - sigma_t * x_0
    return (x_t, v_t)
  
  def get_pred_components(self, x_t, t, pred_type, y_pred, clip_denoise=False):
    """
      Based on the pred_type, return the predict components
      x_t: 4D tensor 
      t: 1D integer index tensor, shape=[batch], value=0, 1, 2, 3, ..., N
      pred_type: [noise, image, velocity]
    """
    var_t = tf.gather(self.var_coefs, t)[:,None,None,None]
    sigma_t = tf.gather(self.sigma_coefs, t)[:,None,None,None]
    mu_t = tf.gather(self.mu_coefs, t)[:,None,None,None]
    (pred_noise, pred_image, pred_velocity) = (None, None, None)
    if pred_type=='noise':
      pred_noise = y_pred
      pred_image = (x_t - sigma_t * pred_noise) / (self.eps+mu_t)
      pred_velocity = mu_t * pred_noise - sigma_t * pred_image
    elif pred_type=='image':
      pred_image = y_pred
      pred_noise = (x_t - mu_t * pred_image) / sigma_t
      pred_velocity = mu_t * pred_noise - sigma_t * pred_image
    elif pred_type=='velocity':
      pred_velocity = y_pred
      pred_image = mu_t * x_t - sigma_t * pred_velocity
      pred_noise = (x_t - mu_t * pred_image) / sigma_t
    else:
      raise NotImplementedError

    if clip_denoise:
      pred_image = tf.clip_by_value(pred_image, self.CLIP_MIN, self.CLIP_MAX)
    return (pred_noise, pred_image, pred_velocity)

  def q_reverse_mean_sigma(self, x_0, x_t, t, pred_noise=None):
    """
      Compute the mean and variance of the diffusion posterior q(x_s | x_t, x_0).
      s = t-n, n>=1
      t: 1D integer index tensor: 1, 2, 3, ..., N
    """
    mu_t = tf.gather(self.mu_coefs, t)[:,None,None,None]
    sigma_t = tf.gather(self.sigma_coefs, t)[:,None,None,None]
    rev_mu_ddim_x0 = tf.gather(self.reverse_mu_ddim_x0, t)[:,None,None,None]
    rev_mu_ddim_noise = tf.gather(self.reverse_mu_ddim_noise, t)[:,None,None,None]
    if pred_noise is None:
      pred_noise = (x_t-mu_t * x_0)/sigma_t
    _mean = keras.layers.Add()([rev_mu_ddim_x0*x_0, rev_mu_ddim_noise*pred_noise])
    _sigma = self.ddim_eta * tf.gather(self.reverse_sigma_coefs, t)[:,None,None,None]
    return (_mean, _sigma)

  def p_sample(self, pred_mean, pred_sigma):
    noise = tf.random.normal(shape=pred_mean.shape, dtype=tf.float32)
    x_s = pred_mean + pred_sigma * noise
    return x_s


# Kernel initializer to use
# This is from original DDPM paper
def kernel_init(scale):
  scale = max(scale, 1e-10)
  return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class TimeEmbedding(keras.layers.Layer):
  """
  A custom Keras layer for generating sinusoidal time embeddings.
  Args:
    dim (int): The dimensionality of the time embedding vector.
    **kwargs: Additional keyword arguments for the parent class.
  """
  def __init__(self, dim, **kwargs):
    super().__init__(**kwargs)
    if dim <=0 or dim % 2 != 0:
      raise ValueError("`dim` must be a positive even integer.")
    self.dim = dim
    self.half_dim = dim // 2
    self.emb = tf.exp(
      tf.range(self.half_dim, dtype=tf.float32) * 
      -(tf.math.log(10000.0) / (self.half_dim - 1))
    )
    
  def call(self, inputs):
    """
    Computes the sinusoidal time embedding for the input tensor.
    Args:
      inuts (tf.Tensor): The input tensor of shape (batch_size,).
    Returns:
      tf.Tensor: The sinusoidal time embedding of shape (batch_size, dim).
    """
    inputs = tf.cast(inputs, dtype=tf.float32)
    if len(inputs.shape) != 1:
      raise ValueError("Input tensor must be 1D.")
    
    emb = inputs[:, None] * self.emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb
  
  def get_config(self):
    """
    Returns the configuration of the layer for serialization.
    """
    config = super().get_config()
    config.update({
      "dim": self.dim,
      "half_dim": self.half_dim,
    })
  
  @classmethod
  def from_config(cls, config):
    """
    Creates a layer from its configuration.
    Args:
      config (dict): The configuration dictionary.
    Returns:
      TimeEmbedding: The TimeEmbedding layer instance.
    """
    return cls(**config)


class SpaceToDepthLayer(keras.layers.Layer):
  def __init__(self, block_size, **kwargs):
    super().__init__(**kwargs)
    self.block_size = block_size

  def call(self, inputs):
    return tf.nn.space_to_depth(inputs, self.block_size)

  def get_config(self):
    config = super().get_config()
    config.update({
      "block_size": self.block_size,
    })
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
    config.update({
      "block_size": self.block_size,
    })
    return config
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  


def ResidualBlock(
    width, attention, num_heads, groups, 
    activation_fn=keras.activations.swish,
    ):
  def apply(inputs):
    x, t = inputs
    input_width = x.shape[3]

    if input_width == width:
      residual = x
    else:
      residual = keras.layers.Conv2D(
        width, kernel_size=1, 
        kernel_initializer=kernel_init(1.0),
        )(x)

    temb = activation_fn(t)
    temb = keras.layers.Dense(
      width, 
      kernel_initializer=kernel_init(1.0),
      )(temb)[:, None, None, :]

    x = keras.layers.GroupNormalization(groups=groups)(x)
    x = activation_fn(x)
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same",
      kernel_initializer=kernel_init(1.0),
      )(x)
    x = keras.layers.Add()([x, temb])
    x = keras.layers.GroupNormalization(groups=groups)(x)
    x = activation_fn(x)
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same",
      kernel_initializer=kernel_init(0.0),
      )(x)
    x = keras.layers.Add()([x, residual])
    # check attention
    if attention:
      res_output = x
      x = keras.layers.GroupNormalization(groups=groups)(x)
      x = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=width, attention_axes=(1,2))(x,x)
      x = keras.layers.Add()([x, res_output])
    return x
  return apply


def DownSample(width):
  def apply(x):
    x = keras.layers.Conv2D(width, kernel_size=3, strides=2, 
      padding="same", 
      kernel_initializer=kernel_init(1.0),
      )(x)
    return x
  return apply


def UpSample(width, interpolation="nearest"):
  def apply(x):
    x = keras.layers.UpSampling2D(size=2, interpolation=interpolation)(x)
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same",
      kernel_initializer=kernel_init(1.0),
      )(x)
    return x
  return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
  def apply(inputs):
    temb = keras.layers.Dense(units, activation=activation_fn, 
      kernel_initializer=kernel_init(1.0),
      )(inputs)
    temb = keras.layers.Dense(
      units, 
      kernel_initializer=kernel_init(1.0),
      )(temb)
    return temb
  return apply


def build_model(
  image_size, 
  image_channel, 
  widths, 
  has_attention,
  num_heads=1,
  num_res_blocks=2, 
  norm_groups=8, 
  interpolation="nearest",
  activation_fn=keras.activations.swish,
  block_size=1,
  temb_dim=128,
  #kernel_init=kernel_init,
  ):
  """
    Build UNet model
    image_size: int, size of input image
    image_channel: int, number of channels of input image
    widths: list of int, width of each block
    has_attention: list of bool, whether to use attention in each block
    num_heads: int, number of heads for attention
    num_res_blocks: int, number of residual blocks in each block
    norm_groups: int, number of groups for normalization
    interpolation: str, interpolation method for upsampling
    activation_fn: activation function for the model
  """
  # Check the input parameters
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

  input_shape = (image_size, image_size, image_channel)
  image_input = keras.Input(shape=input_shape, name="image_input")
  time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
  
  if block_size >1 :
    assert image_size%block_size==0
    x = SpaceToDepthLayer(block_size)(image_input)
  else:
    x = image_input
  
  x = keras.layers.Conv2D(
    widths[0],
    kernel_size=(3, 3),
    padding="same",
    kernel_initializer=kernel_init(1.0),
    )(x)
  
  temb = TimeEmbedding(dim=temb_dim)(time_input)
  temb = TimeMLP(units=temb_dim, activation_fn=activation_fn)(temb)

  skips = [x]

  # DownBlock
  for i in range(len(widths)):
    for _ in range(num_res_blocks):
      x = ResidualBlock(
        widths[i], has_attention[i], num_heads=num_heads, 
        groups=norm_groups, activation_fn=activation_fn)([x, temb])
      skips.append(x)

    if i != len(widths)-1:
      x = DownSample(widths[i])(x)
      skips.append(x)

  # MiddleBlock
  x = ResidualBlock(
    widths[-1], has_attention[-1], num_heads=num_heads,
    groups=norm_groups, activation_fn=activation_fn)([x, temb])
  
  x = ResidualBlock(
    widths[-1], False, num_heads=num_heads,
    groups=norm_groups, activation_fn=activation_fn)([x, temb])

  # UpBlock
  for i in reversed(range(len(widths))):
    for _ in range(num_res_blocks + 1):
      x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
      x = ResidualBlock(widths[i], has_attention[i], num_heads=num_heads, 
        groups=norm_groups, activation_fn=activation_fn)([x, temb])
    
    if i != 0:
      x = UpSample(widths[i], interpolation=interpolation)(x)

  # End block
  x = keras.layers.GroupNormalization(groups=norm_groups)(x)
  x = activation_fn(x)
  x = keras.layers.Conv2D(image_channel*(block_size**2), (3, 3), padding="same", 
    kernel_initializer=kernel_init(0.0),
    name="final_conv2d",
    )(x)
  
  if block_size>1:
    x = DepthToSpaceLayer(block_size)(x)
  return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
  def __init__(self, network, ema_network, timesteps, diff_util, ema=0.999):
    super().__init__()
    self.network = network
    self.ema_network = ema_network
    self.timesteps = timesteps
    self.diff_util = diff_util
    self.ema = ema
    self.loss_tracker = keras.metrics.Mean(name='loss')
    self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
    self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
    self.velocity_loss_tracker = keras.metrics.Mean(name="v_loss")
    assert self.diff_util.pred_type in ['noise', 'image', 'velocity'], \
      "pred_type must be one of [noise, image, velocity]"


  @property
  def metrics(self):
    return [self.loss_tracker, 
            self.noise_loss_tracker, 
            self.image_loss_tracker, 
            self.velocity_loss_tracker,
            ]

  @tf.function
  def train_step(self, images):
    batch_size = tf.shape(images)[0]
    # Random sample timesteps uniformly, t is time index tensor
    t = tf.random.uniform(
      minval=1, maxval=self.timesteps+1, shape=(batch_size,), dtype=tf.int64
    )
    
    with tf.GradientTape() as tape:
      # Sample random noise to be added to the images in the batch
      noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
      # Diffuse the images with noise
      (images_t, v_t) = self.diff_util.q_sample(images, t, noises)
      # get model output
      y_pred = self.network([images_t, t], training=True)
      # get predict components
      pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
        images_t, t, self.diff_util.pred_type, y_pred)
      # calculate losses
      noise_loss = self.loss(noises, pred_noise)
      image_loss = self.loss(images, pred_image)
      velocity_loss = self.loss(v_t, pred_velocity)
      if self.diff_util.pred_type=='noise':
        loss = noise_loss
      elif self.diff_util.pred_type=='image':
        loss = image_loss
      elif self.diff_util.pred_type=='velocity':
        loss = velocity_loss
      else:
        raise ValueError("pred_type must be one of [noise, image, velocity]")

    # Get the gradients of the loss with respect to the model's trainable weights
    # update the weights of the network
    gradients = tape.gradient(loss, self.network.trainable_weights)
    self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
    # Update EMA weights
    self._update_ema_weights()

    # Update metrics
    self.loss_tracker.update_state(loss)
    self.noise_loss_tracker.update_state(noise_loss)
    self.image_loss_tracker.update_state(image_loss)
    self.velocity_loss_tracker.update_state(velocity_loss)

    # Return loss values
    return {m.name: m.result() for m in self.metrics}

  @tf.function
  def _update_ema_weights(self):
    # Update the EMA weights of the model
    for ema_weight, weight in zip(self.ema_network.trainable_weights, self.network.trainable_weights):
      ema_weight.assign(ema_weight * self.ema + (1 - self.ema) * weight)

  @tf.function  
  def test_step(self, images):
    batch_size = tf.shape(images)[0]
    t = tf.random.uniform(minval=1, maxval=self.timesteps+1, 
      shape=(batch_size,), dtype=tf.int64)
    
    noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
    (images_t, v_t) = self.diff_util.q_sample(images, t, noises)
    
    # use ema_network for validaiton
    y_pred = self.ema_network([images_t, t], training=False)

    pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
      images_t, t, self.diff_util.pred_type, y_pred)

    noise_loss = self.loss(noises, pred_noise)
    image_loss = self.loss(images, pred_image)
    velocity_loss = self.loss(v_t, pred_velocity)
    if self.diff_util.pred_type=='noise':
      loss = noise_loss
    elif self.diff_util.pred_type=='image':
      loss = image_loss
    elif self.diff_util.pred_type=='velocity':
      loss = velocity_loss
    else:
      loss = None

    self.loss_tracker.update_state(loss)
    self.noise_loss_tracker.update_state(noise_loss)
    self.image_loss_tracker.update_state(image_loss)
    self.velocity_loss_tracker.update_state(velocity_loss)

    return {m.name: m.result() for m in self.metrics}

  def save_model(self, epoch, logs='mylog.txt', savedir=None):
    if savedir is None:
      savedir = './saved_models'
    # Ensure the save directory exists
    os.makedirs(savedir, exist_ok=True)

    epo = str(epoch).zfill(5)
    
    # Trace the EMA network with sample input
    #sample_input_x, sample_input_t = self.network.inputs
    #sample_input = tf.random.normal([1]+list(sample_input_x.shape[1:]))
    #sample_input_t = tf.constant([1], dtype=tf.int64)
    #_ = self.ema_network([sample_input_x, sample_input_t], training=False)

    if epoch%1000==0 and epoch>0:
      ema_weights_path = os.path.join(savedir, f"ema_epoch_{epo}.weights.h5")
      self.ema_network.save_weights(ema_weights_path)
      #print("save ema weights to {}".format(ema_weights_path))
      
      # Save the full model in .pb format (SavedModel format)
      #pb_path = os.path.join(savedir, f"model_epoch_{epo}_pb")
      #try:
      #  tf.keras.models.save_model(self.ema_network, pb_path, save_format='tf')
      #except Exception as e:
      #  print(f"Error saving model in .pb format: {e}")
      #print(f"Model saved in .pb format to {pb_path}")
      
      # Save the model as a frozen graph
      #frozen_graph_path = os.path.join(savedir, f"frozen_model_epoch_{epo}.pb")
      #self._save_frozen_graph(frozen_graph_path)
      #print(f"Frozen graph saved to {frozen_graph_path}")   
    
    # Save the latest EMA weights
    unet_latest_path = os.path.join(savedir, "unet_latest.weights.h5")
    unet_latest_path_ema = os.path.join(savedir, "unet_latest_ema.weights.h5")
    self.network.save_weights(unet_latest_path)
    self.ema_network.save_weights(unet_latest_path_ema)

    #print(f"EMA latest weights saved to {ema_latest_path}")

  @tf.function
  def _save_frozen_graph(self, frozen_graph_path):
    # Get the concrete function from the model
    concrete_func = self.network.signatures.get("serving_default")
    if concrete_func is None:
        # If no signature is defined, create one
        concrete_func = self.network.__call__.get_concrete_function(
            tf.TensorSpec(self.network.inputs[0].shape, self.network.inputs[0].dtype)
        )

    # Convert the concrete function to a frozen graph
    frozen_func = tf.graph_util.convert_variables_to_constants_v2(concrete_func)
    frozen_graph_def = frozen_func.graph.as_graph_def()

    # Save the frozen graph to a .pb file
    with tf.io.gfile.GFile(frozen_graph_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
  
  def generate_images(self, epoch=None, logs=None, 
    savedir='./', num_images=16, clip_denoise=False, 
    gen_inputs=None, _freeze_ini=False, export_interm=False,
    ):
    #
    img_input, _ = self.network.inputs
    print("image input shape = {}".format(img_input.shape))
    _, img_size, _, img_channel = img_input.shape

    if gen_inputs is None:
      # Randomly sample noise (starting point for reverse process)
      _shape = (num_images, img_size, img_size, img_channel)
      samples = tf.random.normal(shape=_shape, dtype=tf.float32)
      #samples = np.random.randn(*_shape).astype(np.float32)
    else:
      samples = gen_inputs

    n_imgs, _h, _w, _ = samples.shape
    print("generating {} images ...".format(n_imgs))
    
    if _freeze_ini:
      ini_samples = tf.identity(samples)
      ini_samples = ini_samples.numpy()
    
    # Sample from the model iteratively
    #
    # Ex.1: reverse_stride=1, reverse sampling on every steps,
    #   t = tf.range(1000, 0, -1) =  [1000, 999, 998, ..., 3, 2, 1]
    #   xs = x_{t-1)
    #   total = 1000 iterations
    #
    # Ex.2: reverse_stride=10, reverse sampling by every 10 steps
    #   tf.range(1000, 0, -10) = [1000, 990, 980, ...,30,20,10]
    #   xs = x_{t-10}
    #   total = 100 iterations
    
    d_output = {}
    mem_log = open(os.path.join(savedir, "mem.log"), 'w') 
    reverse_timeindex = np.arange(self.timesteps, 0, -self.diff_util.reverse_stride)
    assert reverse_timeindex.dtype=='int64'

    for j, t in enumerate(tqdm.tqdm(reverse_timeindex)):
      tt = tf.fill(n_imgs, t)
      y_pred = self.ema_network.predict([samples, tt], verbose=0, batch_size=16)
      pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
        samples, tt, self.diff_util.pred_type, y_pred, 
        clip_denoise=clip_denoise,
        )

      pred_mean, pred_sigma = self.diff_util.q_reverse_mean_sigma(
        pred_image, samples, tt, pred_noise=pred_noise,
        )
      samples = self.diff_util.p_sample(pred_mean, pred_sigma)
      
      if _freeze_ini:
        samples = samples.numpy()
        samples[:, 0:_h//2, 0:_w//2, :] = ini_samples[:,0:_h//2,0:_w//2,:]
        samples = tf.convert_to_tensor(samples)
      
      del pred_mean
      del pred_sigma
      del pred_noise
      del pred_image
      del pred_velocity
      del y_pred
      # fix the bug of memory leakage
      keras.backend.clear_session()
      gc.collect()
      # fetch current/peak GPU memory
      try: 
        mem = tf.config.experimental.get_memory_info("GPU:0")
      except:
        mem = tf.config.experimental.get_memory_info("CPU:0")
      curr_mb = mem['current'] / (1024**2)
      peak_mb = mem['peak'] / (1024**2)
      mem_log.writelines("t={}, curr MB: {}, peak MB: {}\n".format(t, curr_mb, peak_mb))
      
      if export_interm and t%10==0:
        temp_key = "images_t_"+str(t)
        temp_t_images = samples.numpy()
        temp_t_images = np.clip(temp_t_images, -1, 1)
        temp_t_images = 0.5 * (temp_t_images+1)
        # Chop
        #temp_t_images[temp_t_images < 0.01] = 0
        #temp_t_images[temp_t_images > 0.99] = 1
        d_output[temp_key] = temp_t_images
    
    mem_log.close()

    # de-normalisze the output range to (0, 1)
    output_images = samples.numpy()
    output_images = np.clip(output_images, -1, 1)
    output_images = 0.5 * (output_images+1)
    # Chop
    #output_images[output_images < 0.01] = 0
    #output_images[output_images > 0.99] = 1

    # 3. Return generated samples
    ss = "x".join(list(map(str, samples.numpy().shape)))
    eta = str(self.diff_util.ddim_eta)
    revs = str(self.diff_util.reverse_stride)
    output_fn = "ddim_eta" + eta + "_rev" + revs + "_gen_"+ss 
    if not clip_denoise:
      output_fn = output_fn + "_raw"
    output_fn = os.path.join(savedir, output_fn)
    d_output['images']=output_images
    np.savez_compressed(output_fn, **d_output)
    print("Images Generation Done, save to {}".format(output_fn))
    return None


if __name__=="__main__":
  # TODO simple unittest
  pass
