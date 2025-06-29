import logging
import numpy as np
from time import time
from functools import partial
from glob import glob
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import losses

def epe_loss_fn(y_pred, y_true, target_image_ambit, tanh_norm):
    threshold = 0.5 
    pitch = 4.0
    center = target_image_ambit  # output image size : target_image_ambit*2+1
    up_sample_factor = 1
    estimation_order = 2
    _eps = 1.0e-11
    _grid = pitch

    def _func_no_interp(img):
        center_x = int(center)
        center_y = int(center)
        grid = _grid
        a  = img[:, center_y, center_x-1,0]
        b  = img[:, center_y, center_x+1,0]
        c  = img[:, center_y-1, center_x,0]
        d  = img[:, center_y+1, center_x,0]
        e = img[:,center_y,center_x,0]
        h = img[:,center_y-1,center_x-1,0]
        g = img[:,center_y+1,center_x+1,0]

        if estimation_order>1:
            na = img[:, center_y, center_x-2, 0]
            nb = img[:, center_y, center_x+2, 0]
            nc = img[:, center_y-2, center_x, 0]
            nd = img[:, center_y+2, center_x, 0]
            fx = -na/12.0 + a*2.0/3.0-b*2.0/3.0+nb/12.0
            fy = -nc/12.0 + c*2.0/3.0-d*2.0/3.0+nd/12.0
        else:
            fx  = (a-b)/2.0
            fy  = (c-d)/2.0

        fxx = a+b-2.0*e
        fyy = c+d-2.0*e
        fxy = (-a-b-c-d+2*e+h+g)/2.0
        m = tf.sqrt(tf.square(fx) + tf.square(fy))
        df = threshold - e
        curvature = (fxx * tf.square(fy) - 2*fx*fy*fxy + fyy * tf.square(fx)) / (tf.pow(tf.square(fx) + tf.square(fy), 1.5) + _eps)
        slope = m #(tf.multiply(fx,nx) + tf.multiply(fy,ny))
        norm_x = fx / m
        norm_y = fy / m
        epe = df / tf.abs(slope) * grid
        return epe, curvature/grid, slope/grid

    if tanh_norm == True:
        y_pred_epe, y_pred_curvature, y_pred_slope = _func_no_interp((y_pred + 1)*0.5)
        y_true_epe, y_true_curvature, y_true_slope = _func_no_interp((y_true + 1)*0.5)
    else:
        y_pred_epe, y_pred_curvature, y_pred_slope = _func_no_interp(y_pred)
        y_true_epe, y_true_curvature, y_true_slope = _func_no_interp(y_true)

    curvatureError = tf.abs(y_pred_curvature - y_true_curvature)
    slopeError = (y_pred_slope / y_true_slope) - 1

    y_pred_epe = tf.reduce_mean(tf.square(y_pred_epe))
    curvatureError = tf.reduce_mean(tf.square(curvatureError))
    slopeError = tf.reduce_mean(tf.square(slopeError))

    return y_pred_epe, curvatureError, slopeError

def get_loss_fn(loss_type, target_image_ambit=0, tanh_norm=True):
    if loss_type == "bce":
        loss_fn = losses.BinaryCrossentropy(from_logits=True)
    elif loss_type in ("l1", "mae", "mean_absolute_error"):
        loss_fn = losses.MeanAbsoluteError(reduction="auto")
    elif loss_type in ("l2", "mse", "mean_squared_error"):
        loss_fn = losses.MeanSquaredError(reduction="auto")
    elif loss_type == "epe":
        loss_fn = partial(epe_loss_fn, target_image_ambit=target_image_ambit, tanh_norm=tanh_norm)
    return loss_fn

def get_epe(model, valset, target_image_ambit, tanh_norm, trainer_type="fit"):
    epes = []
    curvatureErrors = []
    slopeErrors = []
    if trainer_type != "fit":
        valset = zip(*valset)
    for img_A, real_B in valset:
        fake_B = model.predict(img_A)
        epe, curvatureError, slopeError = epe_loss_fn(fake_B, real_B, target_image_ambit=target_image_ambit, tanh_norm=tanh_norm)
        epes.append(epe)
        curvatureErrors.append(curvatureError)
        slopeErrors.append(slopeError)
    return np.mean(epes), np.mean(curvatureErrors), np.mean(slopeErrors)

def get_psnrs(model, valset, target_image_ambit, tanh_norm, epoch="epoch", trainer_type="fit"):
    psnrs = []
    epes = []
    curvatureErrors = []
    slopeErrors = []
    if trainer_type != "fit":
        valset = zip(*valset)
    for img_A, real_B in valset:
        fake_B = model.predict(img_A)
        epe, curvatureError, slopeError = epe_loss_fn(fake_B, real_B, target_image_ambit=target_image_ambit, tanh_norm=tanh_norm)
        epes.append(epe)
        curvatureErrors.append(curvatureError)
        slopeErrors.append(slopeError)
        if tanh_norm == True:
            # -1~1  ->  0~1
            fake_B = (fake_B + 1) * 0.5
            real_B = (real_B + 1) * 0.5
        psnrs.append(tf.image.psnr(real_B, fake_B, max_val=1))
    return np.mean(psnrs), np.mean(epes), np.mean(curvatureErrors), np.mean(slopeErrors)

def get_psnrs_adrs0(model, valset, target_image_ambit, tanh_norm, epoch="epoch", trainer_type="fit"):
    psnrs = []
    adrs = []
    epes = []
    curvatureErrors = []
    slopeErrors = []
    if trainer_type != "fit":
        valset = zip(*valset)
    for img_A, real_B in valset:
        fake_B = model.predict(img_A)
        epe, curvatureError, slopeError = epe_loss_fn(fake_B, real_B, target_image_ambit=target_image_ambit, tanh_norm=tanh_norm)
        epes.append(epe)
        curvatureErrors.append(curvatureError)
        slopeErrors.append(slopeError)
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
    return np.mean(psnrs), np.mean(adrs), np.mean(epes), np.mean(curvatureErrors), np.mean(slopeErrors)

def get_psnrs_adrs(model, valset, target_image_ambit, tanh_norm, trainer_type="fit"):
    psnrs = []
    adrs = []
    epes = []
    curvatureErrors = []
    slopeErrors = []

    if trainer_type != "fit":
        valset = zip(*valset)
    for img_A, real_B in valset:
        fake_B = model.predict(img_A)
        epe, curvatureError, slopeError = epe_loss_fn(fake_B, real_B, target_image_ambit=target_image_ambit, tanh_norm=tanh_norm)
        epes.append(epe)
        curvatureErrors.append(curvatureError)
        slopeErrors.append(slopeError)
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
    
    return np.mean(psnrs), np.mean(adrs), np.mean(epes), np.mean(curvatureErrors), np.mean(slopeErrors)


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
        val_epoch,
        img_training_target_ambit,
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
        self.val_epoch=val_epoch
        self.img_training_target_ambit = img_training_target_ambit
        #
        self.min_epe = 123456789
        self.min_epoch = -1

    def on_train_begin(self, logs=None):
        logging.info("----- INITIAL METRICS -----")
        psnr_1, adr_1, epe_1, curvatureErr_1, slopeErr_1 = get_psnrs_adrs(self.model, self.valset_1, target_image_ambit=self.img_training_target_ambit, tanh_norm=self.tanh_norm)
        psnr_2, adr_2, epe_2, curvatureErr_2, slopeErr_2 = get_psnrs_adrs(self.model, self.valset_2, target_image_ambit=self.img_training_target_ambit, tanh_norm=self.tanh_norm)
        logging.info(
            f"valset_EPE: {epe_1} valset_curvErr: {curvatureErr_1} valset_slopeErr: {slopeErr_1} valset_PSNR: {psnr_1} valset_ADR: {adr_1}\n"
            f"trainset_EPE: {epe_2} trainset_curvErr: {curvatureErr_2} trainset_slopeErr: {slopeErr_2} trainset_PSNR: {psnr_2} trainset_ADR: {adr_2}"
        )
        logging.info("--------------------------")

        self.train_start_time = time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            f"[ Epoch {epoch} "
            f"Time: {time() - self.epoch_start_time} ]"
        )
        #
        if (epoch + 1) % self.val_epoch == 0:
            psnr_1, adr_1, epe_1, curvatureErr_1, slopeErr_1 = get_psnrs_adrs(self.model, self.valset_1, target_image_ambit=self.img_training_target_ambit, tanh_norm=self.tanh_norm)
            psnr_2, adr_2, epe_2, curvatureErr_2, slopeErr_2 = get_psnrs_adrs(self.model, self.valset_2, target_image_ambit=self.img_training_target_ambit, tanh_norm=self.tanh_norm)
            logging.info(
                f"valset_EPE: {epe_1} valset_curvErr: {curvatureErr_1} valset_slopeErr: {slopeErr_1} valset_PSNR: {psnr_1} valset_ADR: {adr_1}\n"
                f"trainset_EPE: {epe_2} trainset_curvErr: {curvatureErr_2} trainset_slopeErr: {slopeErr_2} trainset_PSNR: {psnr_2} trainset_ADR: {adr_2}"
            )

            if epe_1 < self.min_epe:
                self.min_epe = epe_1
                self.min_epoch = epoch
                save_model(self.model, self.model.optimizer, self.logging_path, self.save_type, "max")

        save_model(self.model, self.model.optimizer, self.logging_path, self.save_type, "latest")
    
    def on_train_end(self, logs=None):
        logging.info(
            f"----- Train Time: {time() - self.train_start_time} -----"
        )
        logging.info(
            f"----- Min epe: {self.min_epe} epoch {self.min_epoch}-----"
        )