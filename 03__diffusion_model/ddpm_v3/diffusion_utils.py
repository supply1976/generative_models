import numpy as np
import tensorflow as tf
from tensorflow import keras


class DiffusionUtility:
    """
    Utility class for diffusion model computations.
    
    Implements the mathematical operations for forward and reverse diffusion processes,
    supporting multiple noise schedules and prediction types.
    
    The forward process gradually adds noise: q(x_t | x_0) = N(√α_t x_0, (1-α_t)I)
    The reverse process removes noise: p(x_{t-1} | x_t) using learned predictions.
    
    Args:
        b0 (float): Starting noise schedule parameter
        b1 (float): Ending noise schedule parameter  
        scheduler (str): Noise schedule type ('linear', 'cosine', 'cos6')
        timesteps (int): Number of diffusion timesteps
        pred_type (str): Type of model prediction ('noise', 'image', 'velocity')
        reverse_stride (int): Step size for reverse process (1 for DDPM, >1 for DDIM)
        ddim_eta (float): DDIM determinism parameter (0=deterministic, 1=stochastic)
    """

    def __init__(self, b0=0.1, b1=20.0, scheduler='linear', timesteps=1000,
                 pred_type='velocity', reverse_stride=1, ddim_eta=1.0, clip_denoise=True):
        self._validate_params(timesteps, reverse_stride, pred_type)
        self._set_params(b0, b1, scheduler, timesteps, pred_type, reverse_stride, ddim_eta)
        self.timestamps = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
        self.clip_denoise = clip_denoise
        
        # Compute diffusion coefficients
        alphas = self._compute_alphas()
        self._compute_forward_coefficients(alphas)
        self._compute_reverse_coefficients(alphas)
    
    def _validate_params(self, timesteps, reverse_stride, pred_type):
        """Validate initialization parameters."""
        if not isinstance(timesteps, int):
            raise TypeError("timesteps must be an integer")
        if not isinstance(reverse_stride, int):
            raise TypeError("reverse_stride must be an integer")
        if reverse_stride < 1:
            raise ValueError("reverse_stride must be >= 1")
        if timesteps % reverse_stride != 0:
            raise ValueError("timesteps must be divisible by reverse_stride")
        if pred_type not in ['noise', 'image', 'velocity']:
            raise ValueError(f"pred_type must be one of ['noise', 'image', 'velocity'], got {pred_type}")
    
    def _set_params(self, b0, b1, scheduler, timesteps, pred_type, reverse_stride, ddim_eta):
        """Set instance parameters."""
        self.b0 = b0
        self.b1 = b1
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.pred_type = pred_type
        self.reverse_stride = reverse_stride
        self.ddim_eta = ddim_eta
        self.eps = 1.0e-6
        self.CLIP_MIN = -1.0
        self.CLIP_MAX = 1.0
    
    def _compute_alphas(self):
        """Compute alpha values based on the chosen scheduler."""
        if self.scheduler == 'linear':
            Bt = self.timestamps * self.b0 + 0.5 * self.timestamps**2 * (self.b1 - self.b0)
            if not np.all(Bt >= 0):
                raise ValueError("Beta values must be non-negative")
            return np.exp(-Bt)
        
        elif self.scheduler == 'cosine':
            end_angle = 89  # degrees
            angles = self.timestamps * end_angle * np.pi / 180
            return np.cos(angles) ** 2
        
        elif self.scheduler == 'cos6':
            end_angle = 80  # degrees
            angles = self.timestamps * end_angle * np.pi / 180
            return np.cos(angles) ** 6
        
        else:
            raise ValueError(f"Unsupported scheduler '{self.scheduler}'. "
                           f"Supported: ['linear', 'cosine', 'cos6']")
    
    def _compute_forward_coefficients(self, alphas):
        """Compute coefficients for forward diffusion process."""
        mu_coefs = np.sqrt(alphas)
        var_coefs = 1.0 - alphas
        sigma_coefs = np.sqrt(var_coefs)
        
        self.mu_coefs = tf.constant(mu_coefs, tf.float32)
        self.var_coefs = tf.constant(var_coefs, tf.float32)
        self.sigma_coefs = tf.constant(sigma_coefs, tf.float32)
    
    def _compute_reverse_coefficients(self, alphas):
        """Compute coefficients for reverse diffusion process."""
        # Extract alpha values for reverse process
        alpha_t = alphas[self.reverse_stride:]
        alpha_s = alphas[:-self.reverse_stride]
        alpha_ts = alpha_t / alpha_s
        
        # Compute variance coefficients
        var_coefs_st = (1 - alpha_s) / (1 - alpha_t)
        reverse_var_coefs = var_coefs_st * (1 - alpha_ts)
        
        if not np.all(reverse_var_coefs >= 0):
            raise ValueError("Reverse variance coefficients must be non-negative")
        
        reverse_sigma_coefs = np.sqrt(reverse_var_coefs)
        
        # Compute mean coefficients for DDPM and DDIM
        reverse_mu_ddpm_xt = var_coefs_st * np.sqrt(alpha_ts)
        reverse_mu_ddpm_x0 = np.sqrt(alpha_s) * (1 - alpha_ts) / (1 - alpha_t)
        reverse_mu_ddim_x0 = np.sqrt(alpha_s)
        reverse_mu_ddim_noise = np.sqrt(1 - alpha_s - self.ddim_eta * reverse_var_coefs)
        
        # Add padding for first few timesteps
        padding = [0.0] * self.reverse_stride
        reverse_sigma_coefs = np.insert(reverse_sigma_coefs, 0, padding)
        reverse_mu_ddpm_xt = np.insert(reverse_mu_ddpm_xt, 0, padding)
        reverse_mu_ddpm_x0 = np.insert(reverse_mu_ddpm_x0, 0, [1.0] * self.reverse_stride)
        reverse_mu_ddim_x0 = np.insert(reverse_mu_ddim_x0, 0, [1.0] * self.reverse_stride)
        reverse_mu_ddim_noise = np.insert(reverse_mu_ddim_noise, 0, padding)
        
        # Convert to TensorFlow constants
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

    def get_pred_components(self, x_t, t, pred_type, y_pred, clip_denoise):
        """
        Convert model prediction to noise, image, and velocity components.
        
        Args:
            x_t: Noisy image at timestep t
            t: Timestep tensor
            pred_type: Type of prediction ('noise', 'image', 'velocity')
            y_pred: Model prediction
        
        Returns:
            tuple: (pred_noise, pred_image, pred_velocity)
        """
        # Get coefficients for timestep t
        var_t = tf.gather(self.var_coefs, t)[:, None, None, None]
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        
        # Convert prediction based on type
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
            raise ValueError(f"Invalid pred_type: {pred_type}")

        # Optional clipping
        if clip_denoise:
            pred_image = tf.clip_by_value(pred_image, self.CLIP_MIN, self.CLIP_MAX)
        
        return pred_noise, pred_image, pred_velocity

    def q_reverse_mean_sigma(self, x_0, x_t, t, pred_noise=None):
        mu_t = tf.gather(self.mu_coefs, t)[:, None, None, None]
        sigma_t = tf.gather(self.sigma_coefs, t)[:, None, None, None]
        rev_mu_ddim_x0 = tf.gather(self.reverse_mu_ddim_x0, t)[:, None, None, None]
        rev_mu_ddim_noise = tf.gather(self.reverse_mu_ddim_noise, t)[:, None, None, None]
        if pred_noise is None:
            pred_noise = (x_t - mu_t * x_0) / sigma_t
        _mean = rev_mu_ddim_x0 * x_0 + rev_mu_ddim_noise * pred_noise
        _sigma = self.ddim_eta * tf.gather(self.reverse_sigma_coefs, t)[:, None, None, None]
        return (_mean, _sigma)

    def p_sample(self, pred_mean, pred_sigma):
        noise = tf.random.normal(shape=pred_mean.shape, dtype=tf.float32)
        x_s = pred_mean + pred_sigma * noise
        return x_s

    def get_timestep_info(self, t):
        """Get diffusion coefficients for a given timestep."""
        return {
            'mu': tf.gather(self.mu_coefs, t),
            'sigma': tf.gather(self.sigma_coefs, t),
            'var': tf.gather(self.var_coefs, t),
        }

    def validate_timestep(self, t):
        """Validate that timestep is within valid range."""
        if tf.reduce_any(t < 0) or tf.reduce_any(t > self.timesteps):
            raise ValueError(f"Timestep must be in range [0, {self.timesteps}]")

    @property
    def config(self):
        """Return configuration dictionary for serialization."""
        return {
            'b0': self.b0,
            'b1': self.b1,
            'scheduler': self.scheduler,
            'timesteps': self.timesteps,
            'pred_type': self.pred_type,
            'reverse_stride': self.reverse_stride,
            'ddim_eta': self.ddim_eta,
        }

