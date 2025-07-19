import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tqdm
from image_generator import ImageGenerator, MemoryLogger


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, diff_util, num_classes, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.diff_util = diff_util
        self.timesteps = diff_util.timesteps
        self.ema = ema
        self.clip_denoise = diff_util.clip_denoise
        self.num_classes = num_classes
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.velocity_loss_tracker = keras.metrics.Mean(name="v_loss")
        assert self.diff_util.pred_type in ['noise', 'image', 'velocity'], \
            "pred_type must be one of [noise, image, velocity]"
        # Initialize generation components
        self.image_generator = ImageGenerator(self)

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
                images_t, t, self.diff_util.pred_type, y_pred, clip_denoise=True,
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
            images_t, t, self.diff_util.pred_type, y_pred, clip_denoise=True,
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

    def save_ema_model(self, epoch, logs='mylog.txt', savedir=None):
        if savedir is None:
            savedir = './saved_models'
        os.makedirs(savedir, exist_ok=True)
        epo = str(epoch+1).zfill(5)
        output_name = "unet_tf" + tf.__version__ + "ema_"
        if (epoch+1) % 100 == 0 and epoch > 0:
            path_unet_ema_epo = os.path.join(savedir, output_name+f"epoch_{epo}")
            self.ema_network.save(path_unet_ema_epo + ".h5", include_optimizer=False)
            
        path_unet_ema_latest = os.path.join(savedir, output_name+"latest")
        self.ema_network.save(path_unet_ema_latest + ".h5", include_optimizer=False)

    # Convenience methods that delegate to the image generator
    def sample_images(self, **kwargs):
        """Generate samples using the ImageGenerator."""
        return self.image_generator.sample_images(**kwargs)
    
    def generate_images_and_save(self, **kwargs):
        """Generate and save images using ImageGenerator."""
        # Let ImageGenerator handle its own memory logging
        output_dict = self.image_generator.generate_images_and_save(**kwargs)
        
        return output_dict
