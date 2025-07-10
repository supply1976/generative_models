import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, diff_util, ema=0.999,
                 num_classes=None):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.diff_util = diff_util
        self.ema = ema
        self.num_classes = num_classes
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.velocity_loss_tracker = keras.metrics.Mean(name="v_loss")
        assert self.diff_util.pred_type in ['noise', 'image', 'velocity'], \
            "pred_type must be one of [noise, image, velocity]"

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.noise_loss_tracker,
            self.image_loss_tracker,
            self.velocity_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        if isinstance(data, (list, tuple)):
            images, labels = data
        else:
            images, labels = data, None
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(
            minval=1, maxval=self.timesteps + 1, shape=(batch_size,), dtype=tf.int32)
        with tf.GradientTape() as tape:
            noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            images_t, v_t = self.diff_util.q_sample(images, t, noises)
            inputs = [images_t, t]
            if labels is not None:
                inputs.append(labels)
            y_pred = self.network(inputs, training=True)
            pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
                images_t, t, self.diff_util.pred_type, y_pred
            )
            noise_loss = self.loss(noises, pred_noise)
            image_loss = self.loss(images, pred_image)
            velocity_loss = self.loss(v_t, pred_velocity)
            if self.diff_util.pred_type == 'noise':
                loss = noise_loss
            elif self.diff_util.pred_type == 'image':
                loss = image_loss
            elif self.diff_util.pred_type == 'velocity':
                loss = velocity_loss
            else:
                raise ValueError("pred_type must be one of [noise, image, velocity]")

        gradients = tape.gradient(loss, self.network.trainable_weights)
        clipped_grads = [tf.clip_by_norm(g, clip_norm=1.0) if g is not None else None for g in gradients]
        self.optimizer.apply_gradients(zip(clipped_grads, self.network.trainable_weights))

        self._update_ema_weights()

        self.loss_tracker.update_state(loss)
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.velocity_loss_tracker.update_state(velocity_loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def _update_ema_weights(self):
        for ema_weight, weight in zip(self.ema_network.trainable_weights, self.network.trainable_weights):
            ema_weight.assign(ema_weight * self.ema + (1 - self.ema) * weight)

    @tf.function
    def test_step(self, data):
        if isinstance(data, (list, tuple)):
            images, labels = data
        else:
            images, labels = data, None
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(minval=1, maxval=self.timesteps + 1, shape=(batch_size,), dtype=tf.int32)
        noises = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        images_t, v_t = self.diff_util.q_sample(images, t, noises)
        inputs = [images_t, t]
        if labels is not None:
            inputs.append(labels)
        y_pred = self.ema_network(inputs, training=False)
        pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
            images_t, t, self.diff_util.pred_type, y_pred
        )
        noise_loss = self.loss(noises, pred_noise)
        image_loss = self.loss(images, pred_image)
        velocity_loss = self.loss(v_t, pred_velocity)
        if self.diff_util.pred_type == 'noise':
            loss = noise_loss
        elif self.diff_util.pred_type == 'image':
            loss = image_loss
        elif self.diff_util.pred_type == 'velocity':
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
        os.makedirs(savedir, exist_ok=True)
        epo = str(epoch).zfill(5)
        output_name = "unet_tf" + tf.__version__ + "ema_"
        if epoch % 100 == 0 and epoch > 0:
            path_unet_ema_epo = os.path.join(savedir, output_name+f"epoch_{epo}")
            self.ema_network.save(path_unet_ema_epo + ".h5", include_optimizer=False)
            
        path_unet_ema_latest = os.path.join(savedir, output_name+"latest")
        self.ema_network.save(path_unet_ema_latest + ".h5", include_optimizer=False)
        #self.ema_network.save_weights(path_unet_ema_latest + ".weights.h5")

    @tf.function
    def _save_frozen_graph(self, frozen_graph_path):
        concrete_func = self.network.signatures.get("serving_default")
        if concrete_func is None:
            concrete_func = self.network.__call__.get_concrete_function(
                tf.TensorSpec(self.network.inputs[0].shape, self.network.inputs[0].dtype)
            )
        frozen_func = tf.graph_util.convert_variables_to_constants_v2(concrete_func)
        frozen_graph_def = frozen_func.graph.as_graph_def()
        with tf.io.gfile.GFile(frozen_graph_path, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())

    #@tf.function
    def _denoise_step(self, samples, t, clip_denoise, labels=None):
        """Run one reverse diffusion step."""
        tt = tf.fill((tf.shape(samples)[0],), t)
        inputs = [samples, tt]
        if labels is not None:
            inputs.append(labels)
        y_pred = self.ema_network.predict(inputs, batch_size=16, verbose=0)
        pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
            samples, tt, self.diff_util.pred_type, y_pred, clip_denoise=clip_denoise
        )
        pred_mean, pred_sigma = self.diff_util.q_reverse_mean_sigma(
            pred_image, samples, tt, pred_noise=pred_noise
        )
        return self.diff_util.p_sample(pred_mean, pred_sigma)

    def sample_images(
        self,
        num_images=20,
        clip_denoise=True,
        gen_inputs=None,
        labels=None,
    ):
        """Generate ``num_images`` samples and return them as numpy arrays.

        This is a lightweight variant of :meth:`generate_images` used for
        inline evaluation where we only need the final samples rather than
        saving them to disk.
        """
        img_input = self.network.inputs[0]
        _, img_size, _, img_channel = img_input.shape
        if gen_inputs is None:
            _shape = (num_images, img_size, img_size, img_channel)
            samples = tf.random.normal(shape=_shape, dtype=tf.float32)
        else:
            samples = gen_inputs
        if labels is None and self.num_classes is not None:
            labels = tf.random.uniform((num_images,), maxval=self.num_classes, dtype=tf.int32)
        reverse_timeindex = np.arange(
            self.timesteps, 0, -self.diff_util.reverse_stride, dtype=np.int32
        )
        for t in reverse_timeindex:
            samples = self._denoise_step(
                samples, tf.constant(t, dtype=tf.int32), clip_denoise, labels
            )
        output_images = samples.numpy()
        output_images = np.clip(output_images, -1, 1)
        output_images = 0.5 * (output_images + 1)
        return output_images

    def generate_images(
        self,
        epoch=None,
        logs=None,
        savedir='./',
        num_images=20,
        clip_denoise=True,
        gen_inputs=None,
        labels=None,
        _freeze_ini=False,
        export_interm=False,
    ):
        img_input = self.network.inputs[0]
        print("image input shape = {}".format(img_input.shape))
        _, img_size, _, img_channel = img_input.shape
        if gen_inputs is None:
            _shape = (num_images, img_size, img_size, img_channel)
            samples = tf.random.normal(shape=_shape, dtype=tf.float32)
        else:
            samples = gen_inputs
        if labels is None and self.num_classes is not None:
            labels = tf.random.uniform((num_images,), maxval=self.num_classes, dtype=tf.int32)
        n_imgs, _h, _w, _ = samples.shape
        print("generating {} images ...".format(n_imgs))
        if _freeze_ini:
            ini_samples = tf.identity(samples)
            ini_samples = ini_samples.numpy()
        d_output = {}
        mem_log = open(os.path.join(savedir, "mem.log"), 'w')
        reverse_timeindex = np.arange(
            self.timesteps, 0, -self.diff_util.reverse_stride, dtype=np.int32
        )
        for j, t in enumerate(tqdm.tqdm(reverse_timeindex)):
            samples = self._denoise_step(samples, tf.constant(t, dtype=tf.int32), clip_denoise, labels)
            if _freeze_ini:
                samples = samples.numpy()
                samples[:, 0:_h//2, 0:_w//2, :] = ini_samples[:, 0:_h//2, 0:_w//2, :]
                samples = tf.convert_to_tensor(samples)
            gc.collect()
            try:
                mem = tf.config.experimental.get_memory_info("GPU:0")
            except Exception:
                mem = tf.config.experimental.get_memory_info("CPU:0")
            curr_mb = mem['current'] / (1024 ** 2)
            peak_mb = mem['peak'] / (1024 ** 2)
            mem_log.write(f"t={t}, curr MB: {curr_mb}, peak MB: {peak_mb}\n")
            if export_interm and t % 10 == 0:
                temp_key = "images_t_" + str(t)
                temp_t_images = samples.numpy()
                temp_t_images = np.clip(temp_t_images, -1, 1)
                temp_t_images = 0.5 * (temp_t_images + 1)
                d_output[temp_key] = temp_t_images
        mem_log.close()
        output_images = samples.numpy()
        output_images = np.clip(output_images, -1, 1)
        output_images = 0.5 * (output_images + 1)
        output_images[output_images < 0.01] = 0
        output_images[output_images > 0.99] = 1
        ss = "x".join(list(map(str, samples.numpy().shape)))
        eta = str(self.diff_util.ddim_eta)
        revs = str(self.diff_util.reverse_stride)
        output_fn = "ddim_eta" + eta + "_rev" + revs + "_gen_" + ss + "_tf" + tf.__version__
        if not clip_denoise:
            output_fn = output_fn + "_raw"
        output_fn = os.path.join(savedir, output_fn)
        d_output['images'] = output_images
        np.savez_compressed(output_fn, **d_output)
        print("Images Generation Done, save to {}".format(output_fn))
        return None

