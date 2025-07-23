import os
import gc
import numpy as np
import tensorflow as tf
import tqdm


class ImageGenerator:
    """
    Handles image generation and sampling for diffusion models.
    Separated from the main DiffusionModel for better modularity.
    """
    
    def __init__(self, diffusion_model):
        """
        Initialize the ImageGenerator with a trained diffusion model.
        use ema_network for inference.
        
        Args:
            diffusion_model: The DiffusionModel instance containing the trained networks
        """
        self.diffusion_model = diffusion_model
        self.diff_util = diffusion_model.diff_util
        self.network = diffusion_model.network
        self.ema_network = diffusion_model.ema_network
        self.timesteps = diffusion_model.timesteps
        self.num_classes = diffusion_model.num_classes
    
    def _get_input_shape(self):
        """Get the input shape from the network."""
        img_input = self.network.inputs[0]
        _, img_size, _, img_channel = img_input.shape
        return img_size, img_channel
    
    def _prepare_labels(self, num_images, labels=None):
        """Prepare class labels for conditional generation."""
        if labels is None and self.num_classes is not None:
            labels = tf.random.uniform((num_images,), maxval=self.num_classes, dtype=tf.int32)
        return labels
    
    @tf.function
    def _denoise_step(self, samples, t, clip_denoise, labels=None):
        """Run one reverse diffusion step."""
        tt = tf.fill((tf.shape(samples)[0],), t)
        inputs = [samples, tt]
        if labels is not None:
            inputs.append(labels)
        
        y_pred = self.ema_network(inputs, training=False)
        pred_noise, pred_image, pred_velocity = self.diff_util.get_pred_components(
            samples, tt, self.diff_util.pred_type, y_pred, clip_denoise=clip_denoise
        )
        pred_mean, pred_sigma = self.diff_util.q_reverse_mean_sigma(
            pred_image, samples, tt, pred_noise=pred_noise
        )
        return self.diff_util.p_sample(pred_mean, pred_sigma)
    
    def _denoise_step_use_predict(self, samples, t, clip_denoise, labels=None):
        """Run one reverse diffusion step using model.predict().
        This is a lightweight variant that uses predict instead of tf.function.
        """
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
    
    def _postprocess_images(self, samples, chopping=True):
        """Convert samples to final output format."""
        output_images = samples.numpy()
        output_images = np.clip(output_images, -1, 1)
        output_images = 0.5 * (output_images + 1)
        if chopping:
            output_images[output_images < 0.01] = 0
            output_images[output_images > 0.99] = 1
        return output_images
    
    def sample_images(self,
                      reverse_stride=10, 
                      num_images=20, 
                      clip_denoise=False, 
                      labels=None,
                      ):
        """
        Generate samples and return them as numpy arrays.
        
        This is a lightweight variant used for inline evaluation where we only 
        need the final samples rather than saving them to disk.
        
        Args:
            reverse_stride: Step size for reverse diffusion
            num_images: Number of images to generate
            clip_denoise: Whether to clip denoising predictions
            gen_inputs: Optional initial samples (if None, uses random noise)
            labels: Optional class labels for conditional generation
            
        Returns:
            np.ndarray: Generated images as numpy array
        """
        self.diff_util.reverse_stride = reverse_stride
        alpha = self.diff_util._compute_alphas()
        self.diff_util._compute_reverse_coefficients(alpha)
        # prepare initial samples and labels 
        img_size, img_channel = self._get_input_shape()
        shape = (num_images, img_size, img_size, img_channel)
        samples = tf.random.normal(shape=shape, dtype=tf.float32) 
        labels = self._prepare_labels(num_images, labels)
        
        reverse_timeindex = np.arange(self.timesteps, 0, -reverse_stride, dtype=np.int32)
        
        for t in tqdm.tqdm(reverse_timeindex):
            samples = self._denoise_step(
                samples, tf.constant(t, dtype=tf.int32), clip_denoise, labels
            )
            
        return self._postprocess_images(samples)
    
    
    def generate_images_and_save(self,
                                 logs=None,
                                 gen_task="random",
                                 reverse_stride=10,
                                 savedir='./',
                                 num_images=20,
                                 clip_denoise=False, 
                                 base_images=None,
                                 labels=None,
                                 inpaint_mask=None,
                                 freeze_channel=None,
                                 export_intermediate=False,
                                 enable_memory_logging=False,
                                 memory_log_path=None,
                                 save_to_npz=True,
                                 ):
        """
        Generate images with optional intermediate saving and progress tracking.
        
        Args:
            gen_task:
            reverse_stride:
            savedir:
            num_images: Number of images to generate
            clip_denoise: Whether to clip denoising predictions
            base_images:
            labels: Optional class labels
            inpaint_mask: Optional mask for inpainting (if gen_inputs is provided)
            freeze_channel: Optional channel to freeze for inpainting (if gen_inputs is provided)
            export_intermediate: Whether to export intermediate timesteps
            enable_memory_logging: Whether to log memory usage during generation
            memory_log_path: Path to save memory logs (if enabled)
            save_to_npz:
            
        Returns:
            dict: Dictionary containing generated images and optional intermediate steps
        """
        img_size, img_channel = self._get_input_shape()
        print(f"Input shape: ({img_size}, {img_size}, {img_channel})")
        shape = (num_images, img_size, img_size, img_channel)
        samples = tf.random.normal(shape=shape, dtype=tf.float32) 
        labels = self._prepare_labels(num_images, labels)
        
        print(f"Generating {num_images} images...")
        
        if gen_task == 'channel_inpaint':
            assert base_images is not None
            assert freeze_channel is not None
            if freeze_channel < 0 or freeze_channel >= img_channel:
                raise ValueError(f"freeze_channel must be in range [0, {img_channel - 1}]")
            # Create a mask that zeros out the specified channel
            inpaint_mask = np.ones_like(base_images)
            inpaint_mask[..., freeze_channel] = 0
            inpaint_mask = tf.convert_to_tensor(inpaint_mask)
            #base_images = tf.convert_to_tensor(base_images)
            # update the samples for the input of inpaint generation
            samples = samples * inpaint_mask + base_images * (1 - inpaint_mask)
        # Prepare output dictionary
        output_dict = {}
        if export_intermediate:
            output_dict['intermediate'] = {}
        
        # Setup memory logging if enabled
        if enable_memory_logging and memory_log_path is None:
            memory_log_path = "./mem.log"
        
        self.diff_util.reverse_stride = reverse_stride
        alpha = self.diff_util._compute_alphas()
        self.diff_util._compute_reverse_coefficients(alpha)
        
        # Generation loop
        reverse_timeindex = np.arange(
            self.timesteps, 0, -reverse_stride, dtype=np.int32
        )
        use_predict = True if num_images>100 else False 
        for t in tqdm.tqdm(reverse_timeindex):
            if use_predict:
                samples = self._denoise_step_use_predict(
                    samples, tf.constant(t, dtype=tf.int32), clip_denoise, labels)
                # TODO
                # apply gc.collect()
                # this is workaround to resove memory leak issue in TF 2.12 
                # when iteratively calling model.predict() to do inference
                # calling gc.collect() in everytime step will down-grade the performance
                # but resolve the memory accumulation issue
                gc.collect() 
            else:
                samples = self._denoise_step(
                    samples, tf.constant(t, dtype=tf.int32), clip_denoise, labels)
            # Memory logging
            if enable_memory_logging:
                try:
                    with MemoryLogger(memory_log_path) as logger:
                        logger.log_memory(t)
                except Exception as e:
                    print(f"Memory logging failed: {e}")
            
            if inpaint_mask is not None:
                samples = samples * inpaint_mask + base_images * (1 - inpaint_mask)
            
            # Export intermediate results
            if export_intermediate and t % 10 == 0:
                temp_images = self._postprocess_images(samples, chopping=False)
                output_dict['intermediate'][f't_{t}'] = temp_images
            
        # Final postprocessing
        output_images = self._postprocess_images(samples)
        output_dict['images'] = output_images
        if save_to_npz:
            shape_str = "x".join(map(str, output_images.shape))
            eta = str(self.diff_util.ddim_eta)
            revs = str(self.diff_util.reverse_stride)
        
            filename = f"ddim_eta{eta}_rev{revs}_gen_{shape_str}_tf{tf.__version__}"
            if not clip_denoise:
                filename += "_raw"
            
            filepath = os.path.join(savedir, filename)
            os.makedirs(savedir, exist_ok=True)
            np.savez_compressed(filepath, **output_dict)
            print(f"Images saved to {filepath}.npz")
        
        return output_dict


class MemoryLogger:
    """
    Handles memory usage logging during generation.
    """
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = None
    
    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.log_file = open(self.log_path, 'a')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.close()
    
    def log_memory(self, timestep):
        """Log current memory usage."""
        if self.log_file is None:
            return
            
        try:
            mem = tf.config.experimental.get_memory_info("GPU:0")
        except Exception:
            try:
                mem = tf.config.experimental.get_memory_info("CPU:0")
            except Exception:
                return
                
        curr_mb = mem['current'] / (1024 ** 2)
        peak_mb = mem['peak'] / (1024 ** 2)
        self.log_file.write(f"t={timestep}, curr MB: {curr_mb:.2f}, peak MB: {peak_mb:.2f}\n")
        self.log_file.flush()
