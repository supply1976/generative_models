import os, sys
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras


class GausDiffUtil:
  """
  Gaussian diffusion utility.
  Args:
    beta_start: Start value of the scheduled variance
    beta_end: End value of the scheduled variance
    timesteps: Number of time steps in the forward process
  """

  def __init__(self, beta_start=1e-3, beta_end=0.2, beta_schedule='linear', timesteps=100):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.timesteps = timesteps
    self.beta_schedule = beta_schedule
    self.num_timesteps = int(timesteps)
    self.CLIP_MIN = -1.0
    self.CLIP_MAX = 1.0

    # Define the linear variance schedule, Using float64 for better precision
    if beta_schedule=='const':
      # constant beta
      print("use constant beta schedule, beta constant={}".format(beta_start))
      betas = beta_start * np.ones(timesteps)
    elif beta_schedule=='linear':
      betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
      print("use linear beta schedule, beta = ({},{})".format(beta_start, beta_end))
    elif beta_schedule=='cosine':
      print("use cosine beta schedule, beta = ({},{})".format(beta_start, beta_end))
      betas = beta_start + beta_end*(1-np.cos(0.5*np.pi*np.arange(1,timesteps+1)/timesteps))
    else:
      betas=None
    self.betas = betas
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    self.betas = tf.constant(betas, dtype=tf.float32)
    self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
    self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

    # Calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = tf.constant(
      np.sqrt(alphas_cumprod), dtype=tf.float32)

    self.sqrt_one_minus_alphas_cumprod = tf.constant(
      np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)

    self.log_one_minus_alphas_cumprod = tf.constant(
      np.log(1.0 - alphas_cumprod), dtype=tf.float32)

    self.sqrt_recip_alphas_cumprod = tf.constant(
      np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32)

    self.sqrt_recipm1_alphas_cumprod = tf.constant(
      np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32)

    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

    # Log calculation clipped 
    # because the posterior variance is 0 at the beginning
    # of the diffusion chain
    self.posterior_log_variance_clipped = tf.constant(
      np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32)

    self.posterior_mean_coef1 = tf.constant(
      betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
      dtype=tf.float32)

    self.posterior_mean_coef2 = tf.constant(
      (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
      dtype=tf.float32)

  def _extract(self, a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    
    Args:
      a: Tensor to extract from
      t: Timestep for which the coefficients are to be extracted
      x_shape: Shape of the current batched samples
    """
    batch_size = x_shape[0]
    out = tf.gather(a, t)
    return tf.reshape(out, [batch_size, 1, 1, 1])

  def q_mean_variance(self, x_start, t):
    """
    Extracts the mean, and the variance at current timestep.

    Args:
      x_start: Initial sample (before the first diffusion step)
      t: Current timestep
    """
    x_start_shape = tf.shape(x_start)
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
    variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise):
    """
    Diffuse the data.

    Args:
      x_start: Initial sample (before the first diffusion step)
      t: Current timestep
      noise: Gaussian noise to be added at the current timestep
    Returns:
      Diffused samples at timestep `t`
    """
    x_start_shape = tf.shape(x_start)
    v1 = self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start))
    v1 = v1 * x_start
    v2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
    v2 = v2 * noise
    return v1+v2
        
  def predict_start_from_noise(self, x_t, t, noise):
    x_t_shape = tf.shape(x_t)
    v1 = (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape))
    v1 = v1 * x_t
    v2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape)
    v2 = v2 * noise
    return v1 - v2

  def q_posterior(self, x_start, x_t, t):
    """
    Compute the mean and variance of the 
    diffusion posterior q(x_{t-1} | x_t, x_0).

    Args:
      x_start: Stating point(sample) for the posterior computation
      x_t: Sample at timestep `t`
      t: Current timestep
    Returns:
      Posterior mean and variance at current timestep
    """
    x_t_shape = tf.shape(x_t)
    posterior_mean = (
      self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
      + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t)
    posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
    posterior_log_variance_clipped = self._extract(
      self.posterior_log_variance_clipped, t, x_t_shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
    x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
    if clip_denoised:
      x_recon = tf.clip_by_value(x_recon, self.CLIP_MIN, self.CLIP_MAX)
    
    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
      x_start=x_recon, x_t=x, t=t)
    return model_mean, posterior_variance, posterior_log_variance

  def p_sample(self, pred_noise, x, t, clip_denoised=True):
    """
    Sample from the diffuison model.

    Args:
      pred_noise: Noise predicted by the diffusion model
      x: Samples at a given timestep for which the noise was predicted
      t: Current timestep
      clip_denoised (bool): Whether to clip the predicted noise
        within the specified range or not.
    """
    model_mean, _, model_log_variance = self.p_mean_variance(
      pred_noise, x=x, t=t, clip_denoised=clip_denoised)
    noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
    # No noise when t == 0
    nonzero_mask = tf.reshape(
      1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1])
    return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


# Kernel initializer to use
def kernel_init(scale):
  scale = max(scale, 1e-10)
  return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class AttentionBlock(keras.layers.Layer):
  """
  Applies self-attention.
  Args:
    units: Number of units in the dense layers
    groups: Number of groups to be used for GroupNormalization layer
  """

  def __init__(self, units, groups=8, **kwargs):
    self.units = units
    self.groups = groups
    super().__init__(**kwargs)

    self.norm = keras.layers.GroupNormalization(groups=groups)
    self.query = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
    self.key = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
    self.value = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
    self.proj = keras.layers.Dense(units, kernel_initializer=kernel_init(0.0))

  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    scale = tf.cast(self.units, tf.float32) ** (-0.5)

    inputs = self.norm(inputs)
    q = self.query(inputs)
    k = self.key(inputs)
    v = self.value(inputs)

    attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
    attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

    attn_score = tf.nn.softmax(attn_score, -1)
    attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

    proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
    proj = self.proj(proj)
    return inputs + proj


class TimeEmbedding(keras.layers.Layer):
  def __init__(self, dim, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.half_dim = dim // 2
    self.emb = math.log(10000) / (self.half_dim - 1)
    self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

  def call(self, inputs):
    inputs = tf.cast(inputs, dtype=tf.float32)
    emb = inputs[:, None] * self.emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
  def apply(inputs):
    x, t = inputs
    input_width = x.shape[3]

    if input_width == width:
      residual = x
    else:
      residual = keras.layers.Conv2D(
        width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

    temb = activation_fn(t)
    temb = keras.layers.Dense(
      width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

    x = keras.layers.GroupNormalization(groups=groups)(x)
    x = activation_fn(x)
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same",
      kernel_initializer=kernel_init(1.0))(x)
    x = keras.layers.Add()([x, temb])
    x = keras.layers.GroupNormalization(groups=groups)(x)
    x = activation_fn(x)
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same",
      kernel_initializer=kernel_init(0.0))(x)
    x = keras.layers.Add()([x, residual])
    return x
  return apply


def DownSample(width):
  def apply(x):
    x = keras.layers.Conv2D(width, kernel_size=3, strides=2, 
      padding="same", kernel_initializer=kernel_init(1.0))(x)
    return x
  return apply


def UpSample(width, interpolation="nearest"):
  def apply(x):
    x = keras.layers.UpSampling2D(size=2, interpolation=interpolation)(x)
    x = keras.layers.Conv2D(width, kernel_size=3, padding="same",
      kernel_initializer=kernel_init(1.0))(x)
    return x
  return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
  def apply(inputs):
    temb = keras.layers.Dense(units, activation=activation_fn, 
      kernel_initializer=kernel_init(1.0))(inputs)
    temb = keras.layers.Dense(
      units, kernel_initializer=kernel_init(1.0))(temb)
    return temb
  return apply


def build_model(img_size, img_channel, first_conv_channels, widths, 
  has_attention,
  #attn_resolutions=(16,),
  num_res_blocks=2, 
  norm_groups=8, 
  interpolation="nearest",
  activation_fn=keras.activations.swish):
  #
  input_shape = (img_size, img_size, img_channel)
  image_input = keras.Input(shape=input_shape, name="image_input")
  time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

  x = keras.layers.Conv2D(
    first_conv_channels,
    kernel_size=(3, 3),
    padding="same",
    kernel_initializer=kernel_init(1.0))(image_input)

  temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
  temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

  skips = [x]

  # DownBlock
  for i in range(len(widths)):
    for _ in range(num_res_blocks):
      x = ResidualBlock(
        widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
      #if x.shape[1] in attn_resolutions:
      if has_attention[i]:
        x = AttentionBlock(widths[i], groups=norm_groups)(x)
      skips.append(x)

    if i != len(widths)-1:
      x = DownSample(widths[i])(x)
      skips.append(x)

  # MiddleBlock
  x = ResidualBlock(
    widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
  x = AttentionBlock(widths[-1], groups=norm_groups)(x)
  x = ResidualBlock(
    widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

  # UpBlock
  for i in reversed(range(len(widths))):
    for _ in range(num_res_blocks + 1):
      x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
      x = ResidualBlock(widths[i], 
        groups=norm_groups, activation_fn=activation_fn)([x, temb])
      #if x.shape[1] in attn_resolutions:
      if has_attention[i]:
        x = AttentionBlock(widths[i], groups=norm_groups)(x)
    if i != 0:
      x = UpSample(widths[i], interpolation=interpolation)(x)

  # End block
  x = keras.layers.GroupNormalization(groups=norm_groups)(x)
  x = activation_fn(x)
  x = keras.layers.Conv2D(img_channel, (3, 3), padding="same", 
    kernel_initializer=kernel_init(0.0))(x)
  return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
  def __init__(self, network, ema_network, timesteps, gausdiff_util, ema=0.999):
    super().__init__()
    self.network = network
    self.ema_network = ema_network
    self.timesteps = timesteps
    self.gausdiff_util = gausdiff_util
    self.ema = ema
    
  def train_step(self, images):
    # 1. Get the batch size
    batch_size = tf.shape(images)[0]

    # 2. Sample timesteps uniformly
    t = tf.random.uniform(minval=0, maxval=self.timesteps, 
      shape=(batch_size,), dtype=tf.int64)

    with tf.GradientTape() as tape:
      # 3. Sample random noise to be added to the images in the batch
      noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

      # 4. Diffuse the images with noise
      images_t = self.gausdiff_util.q_sample(images, t, noise)

      # 5. Pass the diffused images and time steps to the network
      pred_noise = self.network([images_t, t], training=True)

      # 6. Calculate the loss
      loss = self.loss(noise, pred_noise)

    # 7. Get the gradients
    gradients = tape.gradient(loss, self.network.trainable_weights)

    # 8. Update the weights of the network
    self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

    # 9. Updates the weight values for the network with EMA weights
    for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
      ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

    # 10. Return loss values
    return {"loss": loss}

  def save_model(self, epoch, logs=None, savedir=None):
    epo = str(epoch+1).zfill(4)
    if epoch%50==0:
      self.ema_network.save_weights(os.path.join(savedir, f"ema_epoch_{epo}.weights.h5"))
    self.ema_network.save_weights(os.path.join(savedir, "ema_best.weights.h5"))

  def generate_images(self, epoch=None, logs=None, savedir='./', num_images=10, 
    freeze_1st=False, given_samples=None, export_intermediate=False):
    img_input, _ = self.network.inputs
    print("image input shape = {}".format(img_input.shape))
    _, img_size, _, img_channel = img_input.shape

    if given_samples is not None:
      samples = given_samples
    else:
      # 1. Randomly sample noise (starting point for reverse process)
      samples = tf.random.normal(
        shape=(num_images, img_size, img_size, img_channel),
        dtype=tf.float32)
    
    ini_samples = tf.identity(samples)

    # 2. Sample from the model iteratively
    print("generating images ...")
    for t in reversed(range(0, self.timesteps)):
      tt = tf.cast(tf.fill(tf.shape(samples)[0], t), dtype=tf.int64)
      if t%10==0:
        verbose_code = 1
        print("time step {}".format(t), samples.shape)
      else:
        verbose_code = 0
      pred_noise = self.ema_network.predict(
        [samples, tt], verbose=verbose_code, batch_size=1)
      samples = self.gausdiff_util.p_sample(
        pred_noise, samples, tt, clip_denoised=True)
      if freeze_1st:
        samples = tf.stack([ini_samples[:,:,:,0], samples[:,:,:,1]], axis=-1)

      if export_intermediate and (t%10==0 or t==1):
        output_fn = os.path.join(savedir, "img_t_"+str(t)+".npz")
        np.savez(output_fn, images=samples.numpy())
    # 3. Return generated samples
    # from (-1,1) to (0,1)
    samples = 0.5*(samples+1)
    ss = "x".join(list(map(str, samples.numpy().shape)))
    output_fn = os.path.join(savedir, "gen_"+ss+".npz")
    np.savez(output_fn, images=samples.numpy())
    return samples

