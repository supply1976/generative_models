import os, sys
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm

tf.debugging.disable_traceback_filtering()


class DiffusionUtility:
  def __init__(self, 
    b0=0.1, b1=20.0, scheduler='linear', timesteps=1000, 
    pred_type='noise', reverse_stride=1, ddim_eta = 1.0,
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

    mu_coefs, var_coefs = (None, None)
    if self.scheduler == 'linear':
      """
      same as original DDPM paper, normalize time to 0 ~ 1
      beta(t) = b0 + (b1-b0)*t, for 0 <= t <= 1
      integrated_beta(t): B(t) = b0*t + 0.5*(b1-b0)*t^2, for 0 <= t <= 1
      alpha(t) === exp(-1*B(t))
      alpha_ts === alpha(t)/alpha(s), for t > s >=0
      for discrete time sampling:
        alpha(s) = alpha(t-n) for n>=1, n is the reverse stride
      """
      Bt = self.timesamps * self.b0 + 0.5* self.timesamps**2 * (self.b1-self.b0)
      # Bt=[B0,B1,B2,...,BN], B0 = 0
      assert np.all(Bt>=0)
      alpha_t = np.exp(-1*Bt)
      deltaBt = Bt[reverse_stride:]-Bt[0:-reverse_stride]
      assert len(Bt)==timesteps+1
      assert len(deltaBt)==len(Bt)-reverse_stride
      assert np.all(deltaBt>=0)
      # alpha_ts === alpha(t) / alpha(s) for t > s >= 0
      alpha_ts = np.exp(-1*deltaBt)

    elif self.scheduler == 'cosine':
      """
      same as iDDPM paper, use cosine scheduler for alpha(t)
      for cosine scheduler, 
      directly define alpha(t) is better than defining beta(t)
      alpha(t) === exp(-1*B(t)) = cos(t*pi/2)  ; for 0 <= t <= 1
        --> alpha(0) = 1, alpha(1) = 0
      B(t) = -1*log(cos(t*pi/2))
        --> B(0)=0 , B(1)=inf
      beta(t) === dB(t)/dt = tan(t*pi/2)
        --> beta(0)=0, beta(1)=inf
      """
      end_angle = 89 ; # degree, for cosine scheduler only, hard-coded
      angles = self.timesamps * end_angle *np.pi / 180
      alpha_t = np.cos(angles)**2
      # alpha_ts === alpha(t) / alpha(s) for t > s >= 0
      alpha_ts = alpha_t[reverse_stride:]/alpha_t[0:-reverse_stride]

    elif self.scheduler == 'cos6':
      """
      experiment only
      """
      end_angle = 80 ; # degree
      angles = self.timesamps * end_angle *np.pi / 180
      alpha_t = np.cos(angles)**6
      # alpha_ts === alpha(t) / alpha(s) for t > s >= 0
      alpha_ts = alpha_t[reverse_stride:]/alpha_t[0:-reverse_stride]
    else:
      print("not supported diffusion scheduler, exit")
      return 
    
    # for forward sampling
    mu_coefs = np.sqrt(alpha_t)
    var_coefs = 1 - alpha_t
    sigma_coefs = np.sqrt(var_coefs)
    # define constant Tensors
    self.mu_coefs = tf.constant(mu_coefs, tf.float32)
    self.var_coefs = tf.constant(var_coefs, tf.float32)
    self.sigma_coefs = tf.constant(sigma_coefs, tf.float32)
    
    # for reverse sampling
    var_coefs_st = var_coefs[0:-reverse_stride]/var_coefs[reverse_stride:]
    reverse_var_coefs = var_coefs_st*(1-alpha_ts)
    assert np.all(reverse_var_coefs >= 0)
    reverse_sigma_coefs = np.sqrt(reverse_var_coefs)

    reverse_mu_ddpm_xt = var_coefs_st*np.sqrt(alpha_ts)
    reverse_mu_ddpm_x0 = mu_coefs[0:-reverse_stride]*(1-alpha_ts)/var_coefs[reverse_stride:]
    reverse_mu_ddim_x0 = mu_coefs[0:-reverse_stride]
    reverse_mu_ddim_e = np.sqrt(var_coefs[0:-reverse_stride]-self.ddim_eta*reverse_var_coefs)
    # insert 0 to make the total length = timesteps+1
    reverse_sigma_coefs = np.insert(reverse_sigma_coefs, 0, [0.0]*reverse_stride)
    reverse_mu_ddpm_xt = np.insert(reverse_mu_ddpm_xt, 0, [0.0]*reverse_stride)
    reverse_mu_ddpm_x0 = np.insert(reverse_mu_ddpm_x0, 0, [1.0]*reverse_stride)
    reverse_mu_ddim_x0 = np.insert(reverse_mu_ddim_x0, 0, [1.0]*reverse_stride)
    reverse_mu_ddim_e = np.insert(reverse_mu_ddim_e, 0, [0.0]*reverse_stride)
    # define constant Tensors
    self.reverse_sigma_coefs = tf.constant(reverse_sigma_coefs, tf.float32)
    self.reverse_mu_ddpm_xt = tf.constant(reverse_mu_ddpm_xt, tf.float32)
    self.reverse_mu_ddpm_x0 = tf.constant(reverse_mu_ddpm_x0, tf.float32)
    self.reverse_mu_ddim_x0 = tf.constant(reverse_mu_ddim_x0, tf.float32)
    self.reverse_mu_ddim_e = tf.constant(reverse_mu_ddim_e, tf.float32)

  def q_sample(self, x_0, t, noise):
    """
      Forward Sampling (noising process)
      x_0: 4D input tensor, shape=[batch, h, w, c]
      t: 1D integer index tensor, shape=[batch], value=0, 1, 2, 3, ..., N
      noise: 4D gaussian random number tensor, shape=[batch, h, w, c]
    """
    sigma_t = tf.gather(self.sigma_coefs, t)[:,None,None,None]
    mu_t = tf.gather(self.mu_coefs, t)[:,None,None,None]
    x_t = mu_t * x_0 + sigma_t * noise
    # velocity
    v_t = mu_t * noise - sigma_t * x_0
    return (x_t, v_t)
  
  def get_pred_components(self, x_t, t, pred_type, y_pred, clip_denoise=False):
    """
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
    elif pred_type=='both':
      pred_image, pred_noise = y_pred
      pred_image = var_t*pred_image + mu_t*(x_t-sigma_t*pred_noise)
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
      method = 'ddim' or 'ddpm'
    """
    mu_t = tf.gather(self.mu_coefs, t)[:,None,None,None]
    sigma_t = tf.gather(self.sigma_coefs, t)[:,None,None,None]

    rev_mu_ddim_0 = tf.gather(self.reverse_mu_ddim_x0, t)[:,None,None,None]
    rev_mu_ddim_e = tf.gather(self.reverse_mu_ddim_e, t)[:,None,None,None]
    if pred_noise is None:
      pred_noise = (x_t-mu_t * x_0)/sigma_t
    _mean = keras.layers.Add()([rev_mu_ddim_0*x_0, rev_mu_ddim_e*pred_noise])
    #_mean = rev_mu_ddim_0 * x_0 + rev_mu_ddim_e * pred_noise
    _sigma = self.ddim_eta * tf.gather(self.reverse_sigma_coefs, t)[:,None,None,None]
    return (_mean, _sigma)

  def p_sample(self, pred_mean, pred_sigma):
    noise = tf.random.normal(shape=pred_mean.shape, dtype=tf.float32)
    x_s = pred_mean + pred_sigma * noise
    return x_s


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
    self.query = keras.layers.Dense(units,
      kernel_initializer=kernel_init(1.0),
      )
    self.key = keras.layers.Dense(units, 
      kernel_initializer=kernel_init(1.0),
      )
    self.value = keras.layers.Dense(units, 
      kernel_initializer=kernel_init(1.0),
      )
    self.proj = keras.layers.Dense(units, 
      kernel_initializer=kernel_init(0.0),
      )

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


class SpaceToDepthLayer(keras.layers.Layer):
  def __init__(self, block_size, **kwargs):
    super().__init__(**kwargs)
    self.block_size = block_size

  def call(self, inputs):
    return tf.nn.space_to_depth(inputs, self.block_size)


class DepthToSpaceLayer(keras.layers.Layer):
  def __init__(self, block_size, **kwargs):
    super().__init__(**kwargs)
    self.block_size = block_size

  def call(self, inputs):
    return tf.nn.depth_to_space(inputs, self.block_size)


def ResidualBlock(width, groups=32, activation_fn=keras.activations.swish):
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
  num_resnet_blocks=2, 
  norm_groups=8, 
  interpolation="nearest",
  activation_fn=keras.activations.swish,
  block_size=1,
  pred_both=False,
  ):
  #
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

  temb = TimeEmbedding(dim=widths[0] * 4)(time_input)
  temb = TimeMLP(units=widths[0] * 4, activation_fn=activation_fn)(temb)

  skips = [x]

  # DownBlock
  for i in range(len(widths)):
    for _ in range(num_resnet_blocks):
      x = ResidualBlock(
        widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
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
    for _ in range(num_resnet_blocks + 1):
      x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
      x = ResidualBlock(widths[i], 
        groups=norm_groups, activation_fn=activation_fn)([x, temb])
      if has_attention[i]:
        x = AttentionBlock(widths[i], groups=norm_groups)(x)
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
    #x = tf.nn.depth_to_space(x, bloc_size)
  if pred_both:
    x = keras.layers.Conv2D(image_channel*2, (3,3), padding='same')(x)
    x = tf.split(x, 2, axis=-1)
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
    assert self.diff_util.pred_type is not None

  @property
  def metrics(self):
    return [self.loss_tracker, 
            self.noise_loss_tracker, 
            self.image_loss_tracker, 
            self.velocity_loss_tracker,
            ]

  def train_step(self, images):
    batch_size = tf.shape(images)[0]
    # Sample timesteps uniformly, t is time index tensor
    t = tf.random.uniform(minval=1, maxval=self.timesteps+1, 
      shape=(batch_size,), dtype=tf.int64)

    with tf.GradientTape() as tape:
      # Sample random noise to be added to the images in the batch
      noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
      # Diffuse the images with noise
      (images_t, v_t) = self.diff_util.q_sample(images, t, noises)

      y_pred = self.network([images_t, t], training=True)

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
      elif self.diff_util.pred_type=='both':
        loss = noise_loss + image_loss
      else:
        loss = None

    # Get the gradients
    gradients = tape.gradient(loss, self.network.trainable_weights)

    # Update the weights of the network
    self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
    self.loss_tracker.update_state(loss)
    self.noise_loss_tracker.update_state(noise_loss)
    self.image_loss_tracker.update_state(image_loss)
    self.velocity_loss_tracker.update_state(velocity_loss)

    # Updates the weight values for the network with EMA weights
    for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
      ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

    # Return loss values
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, images):
    batch_size = tf.shape(images)[0]
    # Sample timesteps uniformly, t is time index tensor
    t = tf.random.uniform(minval=1, maxval=self.timesteps+1, 
      shape=(batch_size,), dtype=tf.int64)

    # Sample random noise to be added to the images in the batch
    noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
    # Diffuse the images with noise
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
    elif self.diff_util.pred_type=='both':
      loss = noise_loss + image_loss
    else:
      loss = None

    self.loss_tracker.update_state(loss)
    self.noise_loss_tracker.update_state(noise_loss)
    self.image_loss_tracker.update_state(image_loss)
    self.velocity_loss_tracker.update_state(velocity_loss)

    return {m.name: m.result() for m in self.metrics}

  def save_model(self, epoch, logs=None, savedir=None):
    epo = str(epoch).zfill(5)
    if epoch%1000==0 and epoch>0:
      self.ema_network.save_weights(os.path.join(savedir, f"ema_epoch_{epo}.weights.h5"))
    self.ema_network.save_weights(os.path.join(savedir, "ema_latest.weights.h5"))

  def generate_images(self, epoch=None, logs=None, 
    savedir='./', num_images=10, _freeze=False, clip_denoise=False, 
    gen_inputs=None, save_ini=False, export_interm=False,
    ):
    #
    img_input, _ = self.network.inputs
    print("image input shape = {}".format(img_input.shape))
    _, img_size, _, img_channel = img_input.shape

    if gen_inputs is None:
      # 1. Randomly sample noise (starting point for reverse process) 
      samples = tf.random.normal(
        shape=(num_images, img_size, img_size, img_channel),
        dtype=tf.float32)
    else:
      samples = gen_inputs

    ini_samples = tf.identity(samples)
    n_imgs, _, _, _ = ini_samples.shape
    
    # 2. Sample from the model iteratively
    print("generating {} images ...".format(n_imgs))
    """
    reverse time index:
    example 1: reverse sampling on every steps,
      t = tf.range(1000, 0, -1) =  [1000, 999, 998, ..., 3, 2, 1]
      xs = x_{t-1)
      total = 1000 iterations
    example 2: reverse sampling with stride 10
      tf.range(1000, 0, -10) = [1000, 990, 980, ...,30,20,10]
      xs = x_{t-10}
      total = 100 iterations
    """
    reverse_timeindex = tf.range(self.timesteps, 0, -self.diff_util.reverse_stride)
    for j, t in enumerate(tqdm.tqdm(reverse_timeindex)):
      tt = tf.cast(tf.fill(tf.shape(samples)[0], t), dtype=tf.int64)
      y_pred = self.ema_network.predict([samples, tt], verbose=0, batch_size=1)
      pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
        samples, tt, self.diff_util.pred_type, y_pred, 
        clip_denoise=clip_denoise,
        )

      pred_mean, pred_sigma = self.diff_util.q_reverse_mean_sigma(
        pred_image, samples, tt, pred_noise=pred_noise,
        )
      
      samples = self.diff_util.p_sample(pred_mean, pred_sigma)
      del pred_noise
      del pred_image
      del pred_velocity
      del y_pred
      if _freeze:
        raise NotImplementedError
        # TODO, not yet completed
        #samples = tf.stack([ini_samples[:,:,:,0], samples[:,:,:,1]], axis=-1)
        #arr_0 = ini_samples.numpy()
        #arr_i = samples.numpy()
        #arr_0[:, 32:-32, 32:-32, :] = arr_i[:, 32:-32, 32:-32, :]
        #samples = tf.convert_to_tensor(arr_0, dtype=tf.float32)

      if export_interm and t.numpy()%10==0:
        output_fn = os.path.join(savedir, "img_t_"+str(t.numpy()))
        if not clip_denoise:
          output_fn = output_fn+"_raw"
        np.savez_compressed(output_fn, images=samples.numpy())
    
    # 3. Return generated samples
    ss = "x".join(list(map(str, samples.numpy().shape)))
    output_fn = "ddim_eta"+str(self.diff_util.ddim_eta)+"_gen_"+ss
    if not clip_denoise:
      output_fn = output_fn + "_raw"
    output_fn = os.path.join(savedir, output_fn)
    d={}
    d['images']=samples.numpy()
    if save_ini:
      d['inputs']=ini_samples.numpy()
    np.savez_compressed(output_fn, **d)
    print("Images Generation Done, save to {}".format(output_fn))
    return samples


if __name__=="__main__":
  # simple unittest
  timesteps = 10
  gdu = DiffusionUtility(b0=0.1, b1=20, timesteps=timesteps)
  batch_size = 10
  arr = 2*np.random.rand(batch_size)-1
  x0 = tf.constant(arr.reshape([batch_size,1,1,1]), tf.float32)

  samples = tf.random.normal(shape=(batch_size,1,1,1), dtype=tf.float32)
  for t in reversed(range(1, timesteps+1)):
    tt = tf.cast(tf.fill(batch_size, t), dtype=tf.int64)
    mean_tt, sigma_tt = gdu.q_reverse_mean_sigma(
      x0, samples, tt, pred_noise=None)

    print(t, np.squeeze(mean_tt.numpy()), np.squeeze(sigma_tt.numpy()))
    samples = gdu.p_sample(mean_tt, sigma_tt)

  print(arr)
