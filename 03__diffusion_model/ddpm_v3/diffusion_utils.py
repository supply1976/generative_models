import numpy as np
import tensorflow as tf
from tensorflow import keras


class DiffusionUtility:
    def __init__(self,
        b0=0.1, b1=20.0, scheduler='linear', timesteps=1000,
        pred_type='velocity', reverse_stride=1, ddim_eta=1.0,
    ):
        self.b0 = b0
        self.b1 = b1
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.timesamps = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
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

        alphas = None
        if self.scheduler == 'linear':
            Bt = self.timesamps * self.b0 + 0.5 * self.timesamps**2 * (self.b1 - self.b0)
            assert np.all(Bt >= 0)
            assert len(Bt) == timesteps + 1
            alphas = np.exp(-1 * Bt)
        elif self.scheduler == 'cosine':
            end_angle = 89  # degree
            angles = self.timesamps * end_angle * np.pi / 180
            alphas = np.cos(angles) ** 2
        elif self.scheduler == 'cos6':
            end_angle = 80
            angles = self.timesamps * end_angle * np.pi / 180
            alphas = np.cos(angles) ** 6
        else:
            print("not supported diffusion scheduler, exit")
            return

        assert alphas is not None
        mu_coefs = np.sqrt(alphas)
        var_coefs = 1.0 - alphas
        sigma_coefs = np.sqrt(var_coefs)
        self.mu_coefs = tf.constant(mu_coefs, tf.float32)
        self.var_coefs = tf.constant(var_coefs, tf.float32)
        self.sigma_coefs = tf.constant(sigma_coefs, tf.float32)

        alpha_t = alphas[reverse_stride:]
        alpha_s = alphas[0:-reverse_stride]
        alpha_ts = alpha_t / alpha_s
        var_coefs_st = (1 - alpha_s) / (1 - alpha_t)
        reverse_var_coefs = var_coefs_st * (1 - alpha_ts)
        assert np.all(reverse_var_coefs >= 0)
        reverse_sigma_coefs = np.sqrt(reverse_var_coefs)

        reverse_mu_ddpm_xt = var_coefs_st * np.sqrt(alpha_ts)
        reverse_mu_ddpm_x0 = np.sqrt(alpha_s) * (1 - alpha_ts) / (1 - alpha_t)
        reverse_mu_ddim_x0 = np.sqrt(alpha_s)
        reverse_mu_ddim_noise = np.sqrt(1 - alpha_s - self.ddim_eta * reverse_var_coefs)
        reverse_sigma_coefs = np.insert(reverse_sigma_coefs, 0, [0.0] * reverse_stride)
        reverse_mu_ddpm_xt = np.insert(reverse_mu_ddpm_xt, 0, [0.0] * reverse_stride)
        reverse_mu_ddpm_x0 = np.insert(reverse_mu_ddpm_x0, 0, [1.0] * reverse_stride)
        reverse_mu_ddim_x0 = np.insert(reverse_mu_ddim_x0, 0, [1.0] * reverse_stride)
        reverse_mu_ddim_noise = np.insert(reverse_mu_ddim_noise, 0, [0.0] * reverse_stride)
        self.reverse_sigma_coefs = tf.constant(reverse_sigma_coefs, tf.float32)
        self.reverse_mu_ddpm_xt = tf.constant(reverse_mu_ddpm_xt, tf.float32)
        self.reverse_mu_ddpm_x0 = tf.constant(reverse_mu_ddpm_x0, tf.float32)
        self.reverse_mu_ddim_x0 = tf.constant(reverse_mu_ddim_x0, tf.float32)
        self.reverse_mu_ddim_noise = tf.constant(reverse_mu_ddim_noise, tf.float32)

    def q_sample(self, x_0, t, noise):
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        x_t = mu_t * x_0 + sigma_t * noise
        v_t = mu_t * noise - sigma_t * x_0
        return (x_t, v_t)

    def get_pred_components(self, x_t, t, pred_type, y_pred, clip_denoise=True):
        var_t = tf.gather(self.var_coefs, t)[:, None, None, None]
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        pred_noise = pred_image = pred_velocity = None
        if pred_type == 'noise':
            pred_noise = y_pred
            pred_image = (x_t - sigma_t * pred_noise) / mu_t
            pred_velocity = mu_t * pred_noise - sigma_t * pred_image
        elif pred_type == 'image':
            pred_image = y_pred
            pred_noise = (x_t - mu_t * pred_image) / sigma_t
            pred_velocity = mu_t * pred_noise - sigma_t * pred_image
        elif pred_type == 'velocity':
            pred_velocity = y_pred
            pred_image = mu_t * x_t - sigma_t * pred_velocity
            pred_noise = (x_t - mu_t * pred_image) / sigma_t
        else:
            raise NotImplementedError

        if clip_denoise:
            pred_image = tf.clip_by_value(pred_image, self.CLIP_MIN, self.CLIP_MAX)
        return (pred_noise, pred_image, pred_velocity)

    def q_reverse_mean_sigma(self, x_0, x_t, t, pred_noise=None):
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        rev_mu_ddim_x0 = tf.gather(self.reverse_mu_ddim_x0, t)[:, None, None, None]
        rev_mu_ddim_noise = tf.gather(self.reverse_mu_ddim_noise, t)[:, None, None, None]
        if pred_noise is None:
            pred_noise = (x_t - mu_t * x_0) / sigma_t
        _mean = rev_mu_ddim_x0 * x_0 + rev_mu_ddim_noise * pred_noise
        _sigma = self.ddim_eta * tf.gather(self.reverse_sigma_coefs, t)[:, None, None, None]
        _sigma = tf.cast(_sigma, _mean.dtype)
        return (_mean, _sigma)

    def p_sample(self, pred_mean, pred_sigma):
        noise = tf.random.normal(shape=pred_mean.shape, dtype=pred_mean.dtype)
        pred_sigma = tf.cast(pred_sigma, pred_mean.dtype)
        x_s = pred_mean + pred_sigma * noise
        return x_s

