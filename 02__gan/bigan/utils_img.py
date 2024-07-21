import logging
import numpy as np
from time import time

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras import losses
def get_loss_fn(loss_type):
    if loss_type == "bce":
        loss_fn = losses.BinaryCrossentropy(from_logits=True)
    elif loss_type in ("l1", "mae", "mean_absolute_error"):
        loss_fn = losses.MeanAbsoluteError(reduction="auto")
    elif loss_type in ("l2", "mse", "mean_squared_error"):
        loss_fn = losses.MeanSquaredError(reduction="auto")
    return loss_fn


def get_metric(model, valset, tanh_norm=True, trainer_type="fit"):
    mses = []
    psnrs = []
    adrs = []
    if trainer_type is not "fit":
        valset = zip(*valset)
    for img_A, real_B in valset:
        fake_B = model.predict(img_A)
        mse = losses.MeanSquaredError(reduction="auto")(fake_B, real_B)
        mses.append(mse)
        if tanh_norm == True:
            # -1~1  ->  0~1
            fake_B = (fake_B + 1) * 0.5
            real_B = (real_B + 1) * 0.5
        psnrs.append(tf.image.psnr(real_B, fake_B, max_val=1))
        #
        area_xor_B = tf.einsum('ijkl->il', tf.abs(real_B - fake_B))
        area_real_B = tf.einsum('ijkl->il', real_B)
        adr = tf.math.reduce_sum( area_xor_B[:,0] / area_real_B[:,0] )*100
        adrs.append(adr)
    return np.mean(psnrs), np.mean(mses), np.mean(adrs)

def save_model(model, optimizer, save_path, save_type, epoch):
    # Save model.
    if save_type == "weights":
        g_save_path = f"{save_path}/G-ep_{epoch}"
        model.save_weights(g_save_path)
    elif save_type == "checkpoint":
        checkpoint_dir = f"{save_path}/"
        checkpoint_prefix = f"{checkpoint_dir}/ep"
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=optimizer,
            generator=model,
        )
        checkpoint.save(file_prefix=checkpoint_prefix)
    elif save_type == "entire":
        g_save_path = f"{save_path}/G-ep_{epoch}"
        model.save(g_save_path)
    else:
        raise ValueError(f"model_type {save_type} not recognized.")


class Monitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        valset_1,
        valset_2,
        logging_path,
        target_image_ambit,
        generator_ambit,
        tanh_norm,
        val_epoch
    ):
        self.valset_1 = valset_1.unbatch().batch(
            1
        )  # NOTE: batch_size = 1 for validation
        self.valset_2 = valset_2.unbatch().batch(1)
        self.logging_path = logging_path
        self.save_type = "weights"
        self.target_image_ambit = target_image_ambit
        self.generator_ambit = generator_ambit
        self.tanh_norm = tanh_norm
        self.val_epoch = val_epoch

        self.max_psnr = -1
        self.max_epoch = -1

    def on_train_begin(self, logs=None):
        self.train_start_time = time()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            f"[ Epoch {epoch} "
            f"Time: {time() - self.epoch_start_time} ]"
        )
        if (epoch + 1) % self.val_epoch == 0:
            psnr_1, mse_1, adr_1 = get_metric(self.model, self.valset_1, tanh_norm=self.tanh_norm)
            psnr_2, mse_2, adr_2 = get_metric(self.model, self.valset_2, tanh_norm=self.tanh_norm)
            
            if psnr_1 > self.max_psnr:
                self.max_psnr = psnr_1
                self.max_epoch = epoch
                save_model(self.model, self.model.optimizer, self.logging_path, self.save_type, "max")
            
            logging.info(
                f"valset_MSE: {mse_1} valset_PSNR: {psnr_1} valset_ADR: {adr_1}\n"
                f"trainset_MSE: {mse_2} trainset_PSNR: {psnr_2} trainset_ADR: {adr_2}"
            )

        save_model(self.model, self.model.optimizer, self.logging_path, self.save_type, "latest")

    def on_train_end(self, logs=None):
        logging.info(
            f"----- Train Time: {time() - self.train_start_time} -----"
        )
        logging.info(
            f"----- Max psnr: {self.max_psnr} epoch {self.max_epoch}-----"
        )
