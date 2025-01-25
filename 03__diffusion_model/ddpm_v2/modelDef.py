import os, sys
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm


class DiffusionUtility:
  def __init__(self, b0=0.1, b1=20, timesteps=100, prev_n_step=1, scheduler='linear'):
    self.b0 = b0
    self.b1 = b1
    self.timesteps = timesteps
    self.scheduler = scheduler
    self.timesamps = np.linspace(0, 1, timesteps+1, dtype=np.float64)
    self.eps = 1.0e-6
    self.CLIP_MIN = -1.0
    self.CLIP_MAX = 1.0
    self.prev_n_step = prev_n_step

    mu_coefs, var_coefs = (None, None)
    if self.scheduler == 'linear':
      # beta(t) = b0 + (b1-b0)*t, for 0 <= t <= 1
      # integrated_beta(t): Bt = b0*t + 0.5*(b1-b0)*t^2, for 0 <= t <= 1
      Bt = self.timesamps * self.b0 + 0.5* self.timesamps**2 * (self.b1-self.b0)
      assert np.all(Bt>=0)
      mu_coefs = np.exp(-0.5*Bt)   ; # sqrt(alpha(t))
      var_coefs = 1 - np.exp(-1*Bt)
      sigma_coefs = np.sqrt(var_coefs) # sqrt(1-alpha(t))
    
    else:
      print("no diffusion scheduler, exit")
      return 

    assert mu_coefs is not None
    assert var_coefs is not None
    
    # for forward use
    self.mu_coefs = tf.constant(mu_coefs, tf.float32)
    self.var_coefs = tf.constant(var_coefs, tf.float32)
    self.sigma_coefs = tf.constant(sigma_coefs, tf.float32)
    
    # for backward use
    # Bt=[B0,B1,B2,...,BN], B0 = 0
    # deltaBt = [B1-B0,B2-B1,B3-B2, ..., BN-B_{N-1}]
    deltaBt = Bt[prev_n_step:]-Bt[0:-prev_n_step]
    reverse_var_coefs = (var_coefs[0:-prev_n_step]/var_coefs[prev_n_step:]) * (1-np.exp(-1*deltaBt))
    assert np.all(reverse_var_coefs >= 0)
    reverse_sigma_coefs = np.sqrt(reverse_var_coefs)
    reverse_mu_coefs_t = (var_coefs[0:-prev_n_step]/var_coefs[prev_n_step:]) * (np.exp(-0.5*deltaBt))
    reverse_mu_coefs_0 = mu_coefs[0:-prev_n_step] * (1-np.exp(-1*deltaBt))/var_coefs[prev_n_step:]
    reverse_sigma_coefs = np.insert(reverse_sigma_coefs, 0, [0.0]*prev_n_step)
    reverse_mu_coefs_t = np.insert(reverse_mu_coefs_t, 0, [0.0]*prev_n_step)
    reverse_mu_coefs_0 = np.insert(reverse_mu_coefs_0, 0, [1.0]*prev_n_step)
    self.reverse_sigma_coefs = tf.constant(reverse_sigma_coefs, tf.float32)
    self.reverse_mu_coefs_t = tf.constant(reverse_mu_coefs_t, tf.float32)
    self.reverse_mu_coefs_0 = tf.constant(reverse_mu_coefs_0, tf.float32)

  def q_sample(self, x_0, t, noise):
    # forward sampling
    # x_0: 4D input tensor
    # t: integer index tensor: 1, 2, 3, ..., N
    # noise: 4D [batch, h, w, c] gaussian random number tensor
    sigma_t = tf.gather(self.sigma_coefs, t)
    mu_t = tf.gather(self.mu_coefs, t)
    x_t = mu_t[:,None,None,None]*x_0 + sigma_t[:,None,None,None] * noise
    return x_t
  
  def x0_estimator(self, x_t, t, pred_noise):
    sigma_t = tf.gather(self.sigma_coefs, t)
    mu_t = tf.gather(self.mu_coefs, t)
    x_0 = (x_t - sigma_t[:,None,None,None] * pred_noise) / mu_t[:,None,None,None]
    return x_0

  def q_reverse_mean_sigma(self, x_0, x_t, t):
    """
    Compute the mean and variance of the 
    diffusion posterior q(x_{t-1} | x_t, x_0).
    t: integer index tensor: 1, 2, 3, ..., N

    """
    c_mu_t = tf.gather(self.reverse_mu_coefs_t, t)
    c_mu_0 = tf.gather(self.reverse_mu_coefs_0, t)
    mean_tn1 = c_mu_0[:,None,None,None]*x_0 + c_mu_t[:,None,None,None] * x_t
    #var_tn1 = tf.gather(self.reverse_var_coefs, t)
    sigma_tn1 = tf.gather(self.reverse_sigma_coefs, t)
    return (mean_tn1, sigma_tn1[:,None,None,None])

  def p_sample(self, pred_mean, pred_sigma):
    noise = tf.random.normal(shape=pred_mean.shape, dtype=tf.float32)
    x_tn1 = pred_mean + pred_sigma * noise
    return x_tn1


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


def build_model(image_size, image_channel, first_channel, widths, 
  has_attention,
  #attn_resolutions=(16,),
  num_resnet_blocks=2, 
  norm_groups=8, 
  interpolation="nearest",
  activation_fn=keras.activations.swish):
  #
  input_shape = (image_size, image_size, image_channel)
  image_input = keras.Input(shape=input_shape, name="image_input")
  time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

  x = keras.layers.Conv2D(
    first_channel,
    kernel_size=(3, 3),
    padding="same",
    kernel_initializer=kernel_init(1.0))(image_input)

  temb = TimeEmbedding(dim=first_channel * 4)(time_input)
  temb = TimeMLP(units=first_channel * 4, activation_fn=activation_fn)(temb)

  skips = [x]

  # DownBlock
  for i in range(len(widths)):
    for _ in range(num_resnet_blocks):
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
    for _ in range(num_resnet_blocks + 1):
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
  x = keras.layers.Conv2D(image_channel, (3, 3), padding="same", 
    kernel_initializer=kernel_init(0.0))(x)
  return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
  def __init__(self, network, ema_network, timesteps, diff_util, ema=0.999):
    super().__init__()
    self.network = network
    self.ema_network = ema_network
    self.timesteps = timesteps
    self.diff_util = diff_util
    self.ema = ema
    self.loss_tracker = keras.metrics.Mean(name="ema_loss")

  @property
  def metrics(self):
    return [self.loss_tracker]

  def train_step(self, images):
    # 1. Get the batch size
    batch_size = tf.shape(images)[0]

    # 2. Sample timesteps uniformly
    # t is time index tensor
    t = tf.random.uniform(minval=1, maxval=self.timesteps+1, 
      shape=(batch_size,), dtype=tf.int64)

    with tf.GradientTape() as tape:
      # 3. Sample random noise to be added to the images in the batch
      noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

      # 4. Diffuse the images with noise
      images_t = self.diff_util.q_sample(images, t, noise)
      #true_mean, true_sigma = self.diff_util.q_reverse_mean_sigma(images, images_t, t)

      # 5. Pass the diffused images and time steps to the network
      pred_noise = self.network([images_t, t], training=True)

      # 6. Calculate the loss
      loss = self.loss(noise, pred_noise)

    # 7. Get the gradients
    gradients = tape.gradient(loss, self.network.trainable_weights)

    # 8. Update the weights of the network
    self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
    self.loss_tracker.update_state(loss)

    # 9. Updates the weight values for the network with EMA weights
    for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
      ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

    # 10. Return loss values
    return {"loss": self.loss_tracker.result()}

  def save_model(self, epoch, logs=None, savedir=None):
    epo = str(epoch).zfill(4)
    if (epoch+1)%500==0:
      self.ema_network.save_weights(os.path.join(savedir, f"ema_epoch_{epo}.weights.h5"))
    self.ema_network.save_weights(os.path.join(savedir, "ema_best.weights.h5"))

  def generate_images(self, epoch=None, logs=None, savedir='./', num_images=10, 
    freeze_1st=False, given_samples=None, export_interm=False):
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
    print("generating {} images ...".format(num_images))
    # reverse time index:
    # example: tf.range(100, 0, -1) = [100,99,...,3,2,1]
    rev_timesamps = tf.range(self.timesteps, 0, -self.diff_util.prev_n_step)
    for j, t in enumerate(tqdm.tqdm(rev_timesamps)):
      tt = tf.cast(tf.fill(tf.shape(samples)[0], t), dtype=tf.int64)
      # model predict noise
      pred_noise = self.ema_network.predict(
        [samples, tt], verbose=0, batch_size=1)
      # from pred_noise to x0_recon
      x0_recon = self.diff_util.x0_estimator(samples, tt, pred_noise)
      pred_mean, pred_sigma = self.diff_util.q_reverse_mean_sigma(x0_recon, samples, tt)
      samples = self.diff_util.p_sample(pred_mean, pred_sigma)

      if freeze_1st:
        samples = tf.stack([ini_samples[:,:,:,0], samples[:,:,:,1]], axis=-1)

      if export_interm and t.numpy()%10==0:
        output_fn = os.path.join(savedir, "img_t_"+str(t.numpy())+".npz")
        np.savez_compressed(output_fn, images=samples.numpy())
    # 3. Return generated samples
    #samples = tf.clip_by_value(samples, -1, 1)
    #samples = 0.5*(samples+1) ; # (-1, 1) -> (0, 1)
    ss = "x".join(list(map(str, samples.numpy().shape)))
    output_fn = os.path.join(savedir, "gen_"+ss+"_raw.npz")
    np.savez_compressed(output_fn, images=samples.numpy())
    return samples


if __name__=="__main__":
  timesteps = 10
  gdu = DiffusionUtility(b0=0.1, b1=20, timesteps=timesteps)
  batch_size = 10
  arr = 2*np.random.rand(batch_size)-1
  x0 = tf.constant(arr.reshape([batch_size,1,1,1]), tf.float32)

  samples = tf.random.normal(shape=(batch_size,1,1,1), dtype=tf.float32)
  for t in reversed(range(1, timesteps+1)):
    tt = tf.cast(tf.fill(batch_size, t), dtype=tf.int64)
    mean_tt, sigma_tt = gdu.q_reverse_mean_sigma(x0, samples, tt)
    print(t, np.squeeze(mean_tt.numpy()), np.squeeze(sigma_tt.numpy()))
    samples = gdu.p_sample(mean_tt, sigma_tt, tt)

  print(arr)
  print(gdu.reverse_mu_coefs_t)
  print(gdu.reverse_mu_coefs_0)
  print(gdu.reverse_sigma_coefs)


