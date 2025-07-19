"""Keras callbacks used for training the diffusion model."""

import os
import logging
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tqdm import tqdm


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay."""

    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=0.0):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        warmup_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        cosine_steps = tf.cast(step - self.warmup_steps, tf.float32)
        cosine_total = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * cosine_steps / cosine_total))
        decayed = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr
        return tf.where(step < self.warmup_steps, warmup_lr, decayed)


class TQDMProgressBar(keras.callbacks.Callback):
    """tqdm based progress bar for model.fit."""

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps_per_epoch = self.params['steps']

    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = tqdm(total=self.steps_per_epoch,
                         desc=f"Epoch {epoch+1}/{self.epochs}",
                         leave=False)

    def on_train_batch_end(self, batch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        gs = int(self.model.optimizer.iterations.numpy())
        self.pbar.set_postfix({'lr': f"{lr:.4e}", 'gs': gs})
        self.pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()


class InlineImageGenerationCallback(keras.callbacks.Callback):
    """Keras callback to generate and save images at the end of every N epochs."""
    
    def __init__(self, reverse_stride=50, period=10, savedir='./inline_gen', num_images=4, labels=None):
        super().__init__()
        self.reverse_stride = reverse_stride
        self.period = period
        self.savedir = savedir
        self.num_images = num_images
        self.labels = labels
        os.makedirs(self.savedir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.period == 0:
            print(f"[Callback] Inline Generating {self.num_images} images at epoch {epoch+1}...")
            images = None
            try:
                images = self.model.sample_images(
                    reverse_stride=self.reverse_stride,
                    num_images=self.num_images,
                    clip_denoise=False,
                    labels=self.labels,
                )
            except Exception as e:
                print(f"[Callback] Inline image generation failed: {e}")
            if images is not None:
                images = [np.concatenate(_, axis=1) for _ in np.split(images, 2, axis=0)]
                images = np.concatenate(images, axis=0)
                images = (images*255.0).astype(np.uint8)
                filename = f"epoch_{str(epoch+1).zfill(5)}.png"
                filepath = f"{self.savedir}/{filename}"
                tf.keras.utils.save_img(filepath, images)
                print(f"[Callback] Images saved to {filepath}")


class InlineEvalCallback(keras.callbacks.Callback):
    """Generate samples during training and compute FID."""

    def __init__(self, valid_ds, eval_interval=1000, savedir=None, patience=3, num_images=16):
        super().__init__()
        self.valid_ds = valid_ds
        self.eval_interval = eval_interval
        self.savedir = savedir
        self.patience = patience
        self.num_images = num_images
        self.best = np.inf
        self.wait = 0
        self.valid_iter = iter(valid_ds)
        self.inception = InceptionV3(include_top=False, pooling='avg',
                                     weights="./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                     input_shape=(299, 299, 3))

    def _calc_fid(self, real, fake):
        real = tf.image.resize(real, (299, 299))
        fake = tf.image.resize(fake, (299, 299))
        real = preprocess_input(real * 255.0)
        fake = preprocess_input(fake * 255.0)
        act1 = self.inception(real, training=False)
        act2 = self.inception(fake, training=False)
        mu1 = tf.reduce_mean(act1, axis=0)
        mu2 = tf.reduce_mean(act2, axis=0)
        x1 = act1 - mu1
        x2 = act2 - mu2
        sigma1 = tf.matmul(x1, x1, transpose_a=True) / tf.cast(tf.shape(act1)[0]-1, tf.float32)
        sigma2 = tf.matmul(x2, x2, transpose_a=True) / tf.cast(tf.shape(act2)[0]-1, tf.float32)
        diff = mu1 - mu2
        s12 = tf.matmul(sigma1, sigma2)
        covmean = sqrtm(s12.numpy())
        covmean = tf.cast(tf.math.real(covmean), tf.float32)
        fid = tf.tensordot(diff, diff, axes=1) + tf.linalg.trace(
            sigma1 + sigma2 - 2.0 * covmean)
        return float(fid.numpy())

    def on_train_batch_end(self, batch, logs=None):
        step = int(self.model.optimizer.iterations.numpy())
        if step == 0 or step % self.eval_interval != 0:
            return
        try:
            real_images = next(self.valid_iter)
            if isinstance(real_images, (list, tuple)):
                real_images, labels_batch = real_images
            else:
                labels_batch = None
        except StopIteration:
            self.valid_iter = iter(self.valid_ds)
            real_images = next(self.valid_iter)
            if isinstance(real_images, (list, tuple)):
                real_images, labels_batch = real_images
            else:
                labels_batch = None
        real_images = (real_images + 1.0) / 2.0
        fake_images = self.model.sample_images(num_images=self.num_images, labels=labels_batch)
        fid_value = self._calc_fid(real_images[:self.num_images], fake_images[:self.num_images])
        logging.info(f"[EVAL] step {step}, FID={fid_value:.6f}")
        if fid_value < self.best:
            self.best = fid_value
            self.wait = 0
            if self.savedir is not None:
                best_path = os.path.join(self.savedir, "best_model.h5")
                self.model.ema_network.save(best_path, include_optimizer=False)
                logging.info(f"[EVAL] best model saved to {best_path}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info("[EVAL] Early stopping triggered")
                self.model.stop_training = True
