import os, sys, argparse
import math, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# use TF >= 2.4
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


(CLIP_MIN, CLIP_MAX) = (-1.0, 1.0)


def augment(img):
    """Flips an image left/right randomly."""
    return tf.image.random_flip_left_right(img)


def resize_and_rescale(img, img_size, rescale=False):
    """Resize the image to the desired size first and then
    rescale the pixel values in the range [CLIP_MIN, CLIP_MAX].

    Args:
        img: Image 3D tensor [height, width, channels]
        img_size: int, Desired image size for resizing
    Returns:
        Resized and rescaled image tensor
    """
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)
    img = tf.image.crop_to_bounding_box(
        img, 
        (height - crop_size)// 2, 
        (width - crop_size) // 2,
        crop_size, 
        crop_size)
    img = tf.cast(img, dtype=tf.float32)
    
    # Resize if img_size is given
    if img_size==height and img_size==width:
        pass
    else:
        img = tf.image.resize(img, size=(img_size, img_size), antialias=True)
    if rescale:
        # Rescale the pixel values
        img = (img/255.0) *(CLIP_MAX-CLIP_MIN) + CLIP_MIN
        img = tf.clip_by_value(img, CLIP_MIN, CLIP_MAX)
    return img


def train_preprocessing(x, img_size, rescale):
    img = x["image"]
    img = resize_and_rescale(img, img_size, rescale)
    img = augment(img)
    return img


class GaussianDiffusion:
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.num_timesteps = int(timesteps)

        # Define the linear variance schedule, Using float64 for better precision
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        self.betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(
            alphas_cumprod_prev, dtype=tf.float32)

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
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod)
        self.posterior_variance = tf.constant(
            posterior_variance, dtype=tf.float32)

        # Log calculation clipped 
        # because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32)

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32)

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (
             1.0 - alphas_cumprod), dtype=tf.float32)

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
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
        """Extracts the mean, and the variance at current timestep.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(
            self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

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
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """
        x_t_shape = tf.shape(x_t)
        posterior_mean = (self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, CLIP_MIN, CLIP_MAX)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffuison model.

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
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        #self.norm = keras.layers.GroupNormalization(groups=groups)
        self.norm = keras.layers.LayerNormalization()
        self.query = keras.layers.Dense(
            units, kernel_initializer=kernel_init(1.0))
        self.key = keras.layers.Dense(
            units, kernel_initializer=kernel_init(1.0))
        self.value = keras.layers.Dense(
            units, kernel_initializer=kernel_init(1.0))
        self.proj = keras.layers.Dense(
            units, kernel_initializer=kernel_init(0.0))

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
        attn_score = tf.reshape(
            attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(
            attn_score, [batch_size, height, width, height, width])

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

        #x = keras.layers.GroupNormalization(groups=groups)(x)
        x = keras.layers.LayerNormalization()(x)
        x = activation_fn(x)
        x = keras.layers.Conv2D(
            width, kernel_size=3, padding="same",
            kernel_initializer=kernel_init(1.0))(x)
        x = keras.layers.Add()([x, temb])
        #x = keras.layers.GroupNormalization(groups=groups)(x)
        x = keras.layers.LayerNormalization()(x)
        x = activation_fn(x)
        x = keras.layers.Conv2D(
            width, kernel_size=3, padding="same",
            kernel_initializer=kernel_init(0.0))(x)
        x = keras.layers.Add()([x, residual])
        return x
    return apply


def DownSample(width):
    def apply(x):
        x = keras.layers.Conv2D(
            width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = keras.layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = keras.layers.Conv2D(
            width, kernel_size=3, padding="same",
            kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = keras.layers.Dense(
            units, 
            activation=activation_fn, 
            kernel_initializer=kernel_init(1.0))(inputs)
        temb = keras.layers.Dense(
            units, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply


def build_model(
    img_size,
    img_channels,
    first_conv_channels,
    widths,
    has_attention,
    #attn_resolutions=(16,),
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish):
    #
    image_input = keras.Input(
        shape=(img_size, img_size, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

    x = keras.layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0))(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(
        units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], 
                groups=norm_groups, activation_fn=activation_fn)([x, temb])
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
            x = ResidualBlock(
                widths[i], 
                groups=norm_groups, activation_fn=activation_fn)([x, temb])
            #if x.shape[1] in attn_resolutions:
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    #x = keras.layers.GroupNormalization(groups=norm_groups)(x)
    x = keras.layers.LayerNormalization()(x)
    x = activation_fn(x)
    x = keras.layers.Conv2D(
        img_channels, (3, 3), 
        padding="same", 
        kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
    
    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, 
            shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    def generate_images(self, num_images=16, given_samples=None):
        img_input, _ = self.network.inputs
        print("image input shape = {}".format(img_input.shape))
        _, img_size, _, img_channels = img_input.shape

        if given_samples is not None:
            samples = given_samples
        else:
            # 1. Randomly sample noise (starting point for reverse process)
            samples = tf.random.normal(
                shape=(num_images, img_size, img_size, img_channels),
                dtype=tf.float32)
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images)
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True)
        # 3. Return generated samples
        return samples

    def plot_images(self, epoch=None, logs=None, 
        num_rows=2, num_cols=8, figsize=(12, 5)):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (
            tf.clip_by_value(
                generated_samples * 127.5 + 127.5, 0.0, 255.0)
                .numpy().astype(np.uint8))

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")
        plt.tight_layout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--restore_model', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--dataset_check', action='store_true')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--total_timesteps', type=int, default=1000)
    parser.add_argument('--norm_groups', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--img_size', required=True, type=int)
    parser.add_argument('--tfdataset', type=str, default=None, 
      help='tf dataset name, available: "oxford_flowers102')
    parser.add_argument('--dbfile', type=str, default=None) 
    parser.add_argument('--first_ch', type=int, default=64)
    parser.add_argument('--ch_mul', nargs='+', type=int, default=[1, 2, 4, 8])

    FLAGS, _ = parser.parse_known_args()
    
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    total_timesteps = FLAGS.total_timesteps
    norm_groups = FLAGS.norm_groups  
    # Number of groups used in GroupNormalization layer
    learning_rate = FLAGS.learning_rate
    img_size = FLAGS.img_size
    first_ch = FLAGS.first_ch
    ch_mul = FLAGS.ch_mul
    widths = [first_ch * mult for mult in ch_mul]
    has_attention = [False, False, True, True]
    assert len(ch_mul)==len(has_attention)
    num_res_blocks = 2  # Number of residual blocks
    
    gpus = tf.config.list_physical_devices("GPU")
    #logical_gpus = tf.config.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    if gpus:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

    dbfile = FLAGS.dbfile
    dbname = "_"
    if FLAGS.tfdataset is not None:
        # Load the dataset, tensorflow_datasets available after TF v2.4
        ds = tfds.load(
            FLAGS.tfdataset, 
            split="train", with_info=False, shuffle_files=False)
        dbname = FLAGS.tfdataset
    else:
        if FLAGS.dbfile is None:
            images = np.load("./random_patterns.npy")
            if len(images.shape)==3:
                images = np.expand_dims(images, axis=-1)
            print("use random patterns {} for flow testing".format(
                images.shape))
            ds = tf.data.Dataset.from_tensor_slices({"image": images})
        elif dbfile.endswith(".npz"):
            data = np.load(dbfile)
            images = data['images']
            if len(images.shape)==3:
                images = np.expand_dims(images, axis=-1)
            print("{} has shape {}".format(dbfile, images.shape))
            ds = tf.data.Dataset.from_tensor_slices({"image": images})
            dbname = "custom"
        else:
            print("not support")
            return
    if FLAGS.tfdataset is None:
        ds = ds.map(lambda x: train_preprocessing(x, img_size, rescale=False))
    else:
        ds = ds.map(lambda x: train_preprocessing(x, img_size, rescale=True))
    
    img_check = [x.numpy() for x in ds.take(1)][0]
    print(img_check.shape)
    h, w, c = img_check.shape
    img_channels = c
    
    # Build the unet model
    network = build_model(
        img_size=img_size, 
        img_channels=img_channels,
        first_conv_channels=first_ch,
        widths=widths,
        #attn_resolutions=(16,),
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish)

    ema_network = build_model(
        img_size=img_size, 
        img_channels=img_channels,
        first_conv_channels=first_ch,
        widths=widths,
        #attn_resolutions=(16,),
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish)

    network.summary()
    ema_network.set_weights(network.get_weights())  
    # Initially the weights are the same

    # Get an instance of the Gaussian Diffusion utilities
    gdf_util = GaussianDiffusion(timesteps=total_timesteps)
    
    net_tag = [str(img_size), str(first_ch), "".join([str(_) for _ in ch_mul])]
    net_tag = ".".join(net_tag)
    model_name = ".".join([dbname, net_tag])

    # restore the trained model weights
    if FLAGS.restore_model and FLAGS.model_dir is not None:
        model_path = os.path.join(FLAGS.model_dir, "ema_best.weights.h5")
        print("load model path: {}".format(model_path))
        load_status = ema_network.load_weights(model_path)
        network.set_weights(ema_network.get_weights())
        #load_status.assert_consumed()

    # Get the model
    model = DiffusionModel(
        network=network, 
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=total_timesteps)
    
    if FLAGS.inference:
        t0 = time.time()
        model.plot_images(num_rows=5, num_cols=10)
        print("inference time = {} seconds".format(time.time() - t0))

    elif FLAGS.training:
        # Compile the model
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
        train_ds = (
            #ds(num_parallel_calls=tf.data.AUTOTUNE)
            ds
            .batch(batch_size, drop_remainder=True)
            .shuffle(batch_size * 2)
            .prefetch(tf.data.AUTOTUNE))

        # Train the model
        savedir = model_name
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,          
            callbacks=[
                keras.callbacks.LambdaCallback(on_train_end=model.plot_images),
                ])
        ema_network.save_weights(os.path.join(savedir, "ema_best.weights.h5"))

    elif FLAGS.dataset_check:
        ds_check = ds.take(16)
        #ds_check = ds_check.map(lambda x: train_preprocessing(x, img_size))
        img_check = np.stack([x.numpy() for x in ds_check], axis=0)
        fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(10,4))
        axes = axes.flatten()
        for i in range(16):
            axes[i].imshow(img_check[i])
            axes[i].axis('off')
        print(
             img_check.shape, img_check.dtype, img_check.min(), img_check.max())
    else:
        print("no action")


if __name__=="__main__":
    main()
    plt.tight_layout()
    plt.show()


