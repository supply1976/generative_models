"""
run.py
------
Main script for training and image generation using diffusion models.

Features:
- Configurable training and image generation via YAML config and command-line flags.
- Efficient dataset loading and prefetching.
- Model checkpointing, logging, and evaluation with FID.
- Supports XLA JIT compilation for performance.

Usage:
    python run.py --config config.yaml --training
    python run.py --config config.yaml --training --enable_xla
    python run.py --config config.yaml --imgen
"""

import os
import time
import datetime
import argparse
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

from diffusion_utils import DiffusionUtility
from layers import TimeEmbedding, TimeMLP, ResidualBlock, DownSample, UpSample, kernel_init
from unet import build_model
from diffusion_model import DiffusionModel
from data_loader import DataLoader
from callbacks import WarmUpCosine, TQDMProgressBar, InlineImageGenerationCallback


# =====================
# Configuration Classes
# =====================

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    path: str
    label_key: Optional[str] = None
    crop_size: Optional[int] = None
    crop_type: str = 'center'
    crop_position: str = 'center'
    augment: bool = False
    augment_type: Optional[str] = None


@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    scheduler: str
    timesteps: int
    num_res_blocks: int
    block_size: int
    norm_groups: int
    first_channel: int
    channel_multiplier: list
    has_attention: list
    num_heads: int
    time_emb_dim: int
    dropout_rate: float = 0.0
    kernel_size: int = 3
    use_cross_attention: bool = False
    num_classes: Optional[int] = None
    class_emb_dim: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    input_image_size: int
    input_image_channel: int
    output_dir: str
    is_new_train: bool
    trained_h5: Optional[str]
    pred_type: str
    loss_fn: str
    epochs: int
    save_period: Optional[int]
    batch_size: int
    steps_per_epoch: Optional[int]
    lr_type: str
    learning_rate: float
    warmup_steps: Optional[int]
    inline_gen_enable: bool = True
    inline_gen_nums: int = 20
    inline_gen_period: int = 10
    inline_gen_reverse_stride: int = 10


@dataclass
class ImageGenConfig:
    """Image generation configuration."""
    model_path: str
    gen_task: str = 'random_uncond'
    num_gen_images: int = 20
    external_npz_input: Optional[str] = None
    class_label: Optional[Union[int, list]] = None
    freeze_channel: Optional[int] = None
    reverse_stride: int = 10
    ddim_eta: float = 1.0
    export_interm: bool = False
    gen_save_dir: Optional[str] = None
    random_seed: Optional[int] = None
    clip_denoise: bool = False
    new_image_size: Optional[int] = None


# =====================
# Core Classes
# =====================

class ConfigManager:
    """Handles configuration parsing and validation."""
    
    @staticmethod
    def parse_config(config_path: str) -> Tuple[DatasetConfig, TrainingConfig, NetworkConfig, ImageGenConfig]:
        """Parse YAML config file into structured configs."""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Parse dataset config
        dataset_dict = cfg['DATASET']
        preprocessing = dataset_dict.get('PREPROCESSING', {})
        dataset_config = DatasetConfig(
            name=dataset_dict['NAME'],
            path=dataset_dict['PATH'],
            label_key=dataset_dict.get('LABEL_KEY'),
            crop_size=preprocessing.get('CROP_SIZE'),
            crop_type=preprocessing.get('CROP_TYPE', 'center'),
            crop_position=preprocessing.get('CROP_POSITION', 'center'),
            augment=preprocessing.get('AUGMENT', False),
            augment_type=preprocessing.get('AUGMENT_TYPE')
        )
        
        # Parse training config
        training_dict = cfg['TRAINING']
        inline_gen = training_dict.get('INLINE_GEN', {})
        hyper_params = training_dict['HYPER_PARAMETERS']
        
        training_config = TrainingConfig(
            input_image_size=training_dict['INPUT_IMAGE_SIZE'],
            input_image_channel=training_dict['INPUT_IMAGE_CHANNEL'],
            output_dir=training_dict['OUTPUT_DIR'],
            is_new_train=training_dict['IS_NEW_TRAIN'],
            trained_h5=training_dict.get('TRAINED_H5'),
            pred_type=training_dict['PRED_TYPE'],
            loss_fn=training_dict['LOSS_FN'],
            epochs=hyper_params['EPOCHS'],
            save_period=hyper_params.get('SAVE_PERIOD'),
            batch_size=hyper_params['BATCH_SIZE'],
            steps_per_epoch=hyper_params.get('STEPS_PER_EPOCH'),
            lr_type=hyper_params['LR_TYPE'],
            learning_rate=hyper_params['LEARNING_RATE'],
            warmup_steps=hyper_params.get('WARMUP_STEPS'),
            inline_gen_enable=inline_gen.get('ENABLE', True),
            inline_gen_nums=inline_gen.get('NUMS', 20),
            inline_gen_period=inline_gen.get('PERIOD', 10),
            inline_gen_reverse_stride=inline_gen.get('REVERSE_STRIDE', 10)
        )
        
        # Parse network config
        network_dict = training_dict['NETWORK']
        network_config = NetworkConfig(
            scheduler=network_dict['SCHEDULER'],
            timesteps=network_dict['TIMESTEPS'],
            num_res_blocks=network_dict['NUM_RES_BLOCKS'],
            block_size=network_dict['BLOCK_SIZE'],
            norm_groups=network_dict['NORM_GROUPS'],
            first_channel=network_dict['FIRST_CHANNEL'],
            channel_multiplier=network_dict['CHANNEL_MULTIPLIER'],
            has_attention=network_dict['HAS_ATTENTION'],
            num_heads=network_dict['NUM_HEADS'],
            time_emb_dim=network_dict['TIME_EMB_DIM'],
            dropout_rate=network_dict.get('DROPOUT_RATE', 0.1),
            kernel_size=network_dict.get('KERNEL_SIZE', 3),
            use_cross_attention=network_dict.get('USE_CROSS_ATTENTION', False),
            num_classes=network_dict.get('NUM_CLASSES'),
            class_emb_dim=network_dict.get('CLASS_EMB_DIM')
        )
        
        # Parse image generation config
        imgen_dict = cfg['IMAGE_GENERATION']
        imgen_config = ImageGenConfig(
            model_path=imgen_dict.get('MODEL_PATH', ''),
            gen_task=imgen_dict.get('GEN_TASK', 'random_uncond'),
            num_gen_images=imgen_dict.get('NUM_GEN_IMAGES', 20),
            external_npz_input=imgen_dict.get('EXTERNAL_NPZ_INPUT'),
            class_label=imgen_dict.get('CLASS_LABEL'),
            freeze_channel=imgen_dict.get('FREEZE_CHANNEL'),
            reverse_stride=imgen_dict.get('REVERSE_STRIDE', 10),
            ddim_eta=imgen_dict.get('DDIM_ETA', 1.0),
            export_interm=imgen_dict.get('EXPORT_INTERM', False),
            gen_save_dir=imgen_dict.get('GEN_SAVE_DIR'),
            random_seed=imgen_dict.get('RANDOM_SEED'),
            clip_denoise=imgen_dict.get('CLIP_DENOISE', False),
            new_image_size=imgen_dict.get('NEW_IMAGE_SIZE'),
        )
        
        return dataset_config, training_config, network_config, imgen_config


class ModelBuilder:
    """Handles model construction and related utilities."""
    
    @staticmethod
    def build_models(image_size: int, image_channel: int, network_config: NetworkConfig):
        """Build main and EMA models."""
        widths = [network_config.first_channel * mult for mult in network_config.channel_multiplier]
        
        kwargs = dict(
            image_size=image_size,
            image_channel=image_channel,
            widths=widths,
            has_attention=network_config.has_attention,
            num_heads=network_config.num_heads,
            num_res_blocks=network_config.num_res_blocks,
            norm_groups=network_config.norm_groups,
            actf=keras.activations.swish,
            block_size=network_config.block_size,
            temb_dim=network_config.time_emb_dim,
            dropout_rate=network_config.dropout_rate,
            kernel_size=network_config.kernel_size,
            use_cross_attention=network_config.use_cross_attention,
            num_classes=network_config.num_classes,
            class_emb_dim=network_config.class_emb_dim,
        )
        
        network = build_model(**kwargs)
        ema_network = build_model(**kwargs)
        ema_network.set_weights(network.get_weights())
        return network, ema_network
    
    @staticmethod
    def create_diffusion_utility(network_config: NetworkConfig, training_config: TrainingConfig, 
                                 reverse_stride: int = 1, ddim_eta: float = 1.0, 
                                 clip_denoise: bool = False) -> DiffusionUtility:
        """Create diffusion utility with common parameters."""
        return DiffusionUtility(
            b0=0.1, 
            b1=20, 
            timesteps=network_config.timesteps,
            scheduler=network_config.scheduler, 
            pred_type=training_config.pred_type, 
            reverse_stride=reverse_stride,
            ddim_eta=ddim_eta, 
            clip_denoise=clip_denoise
        )
    
    @staticmethod
    def create_lr_schedule(training_config: TrainingConfig):
        """Create learning rate schedule."""
        if training_config.lr_type == 'constant':
            return training_config.learning_rate
        elif training_config.lr_type == 'warmup_cosine':
            assert training_config.warmup_steps is not None
            total_steps = (training_config.epochs * training_config.steps_per_epoch 
                          if training_config.steps_per_epoch else None)
            return WarmUpCosine(
                base_lr=training_config.learning_rate,
                warmup_steps=training_config.warmup_steps,
                total_steps=total_steps
            )
        elif training_config.lr_type == 'cosine_decay':
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=training_config.learning_rate,
                decay_steps=10000,
                alpha=0.0
            )
        else:
            raise NotImplementedError(f"Learning rate type {training_config.lr_type} not implemented")
    
    @staticmethod
    def create_loss_function(loss_fn_name: str):
        """Create loss function."""
        if loss_fn_name == "MAE":
            return keras.losses.MeanAbsoluteError()
        elif loss_fn_name == 'MSE':
            return keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError(f"Loss function {loss_fn_name} not implemented")


class DirectoryManager:
    """Handles directory creation and logging setup."""
    
    @staticmethod
    def init_logging(filename: str, checkpoint: Optional[str] = None):
        """Initialize logging to file and console."""
        mode = "w+"
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(filename, mode=mode), logging.StreamHandler()],
        )
    
    @staticmethod
    def setup_training_directories(dataset_config: DatasetConfig, training_config: TrainingConfig, 
                                   network_config: NetworkConfig, config_file: str) -> str:
        """Set up training directories and logging."""
        os.makedirs(training_config.output_dir, exist_ok=True)
        
        # Create model name tag
        model_nametag = f"unet{network_config.first_channel}m{''.join(map(str, network_config.channel_multiplier))}"
        model_nametag += f"g{network_config.norm_groups}rb{network_config.num_res_blocks}bk{network_config.block_size}"
        
        # Determine image size
        input_image_size = dataset_config.crop_size or training_config.input_image_size
        input_shape = (input_image_size, input_image_size, training_config.input_image_channel)
        
        # Create dataset and model directories
        dataset_tag = os.path.join(
            os.path.abspath(training_config.output_dir),
            f"{dataset_config.name}_{input_shape[0]}x{input_shape[1]}x{input_shape[2]}"
        )
        os.makedirs(dataset_tag, exist_ok=True)
        
        tr_output_dir = os.path.join(dataset_tag, model_nametag)
        os.makedirs(tr_output_dir, exist_ok=True)
        
        # Create timestamped directory
        dateID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dateID = "_".join([network_config.scheduler, str(network_config.timesteps), 
                          training_config.pred_type, training_config.loss_fn, dateID])
        
        if training_config.is_new_train:
            logging_dir = os.path.join(tr_output_dir, dateID)
            os.makedirs(logging_dir, exist_ok=True)
            DirectoryManager.init_logging(os.path.join(logging_dir, "train.log"))
            logging.info("[INFO] Start a new training")
        else:
            if not training_config.trained_h5:
                raise ValueError('If "IS_NEW_TRAIN: false", you must provide a trained .h5 file')
            
            restored_model_ID = os.path.basename(os.path.dirname(training_config.trained_h5)).split("_")[-1]
            logging_dir = os.path.join(tr_output_dir, f"{dateID}_from_{restored_model_ID}")
            os.makedirs(logging_dir, exist_ok=True)
            DirectoryManager.init_logging(os.path.join(logging_dir, "train.log"))
            logging.info(f"[INFO] Restoring model from: {training_config.trained_h5}")
            logging.info("[INFO] Continuous Transfer training ...")
        
        # Copy config file
        shutil.copy(config_file, os.path.join(logging_dir, "training_config.yaml"))
        return logging_dir


class DatasetManager:
    """Handles dataset loading and preparation."""
    
    @staticmethod
    def prepare_datasets(dataset_config: DatasetConfig, training_config: TrainingConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare training and validation datasets with efficient prefetching."""
        autotune = tf.data.AUTOTUNE
        input_image_size = dataset_config.crop_size or training_config.input_image_size
        
        if os.path.isdir(dataset_config.path):
            dataloader = DataLoader(
                data_dir=dataset_config.path,
                img_size=input_image_size,
                crop_size=dataset_config.crop_size,
                crop_type=dataset_config.crop_type,
                crop_position=dataset_config.crop_position,
                augment=dataset_config.augment,
                augment_type=dataset_config.augment_type,
                label_key=dataset_config.label_key,
            )
            train_ds, valid_ds = dataloader._get_dataset()
        else:
            raise NotImplementedError("NPZ dataset loading not implemented in refactored version")
        
        # Batch and prefetch
        train_ds = train_ds.batch(training_config.batch_size, drop_remainder=True)
        valid_ds = valid_ds.batch(training_config.batch_size)
        train_ds = train_ds.prefetch(autotune)
        valid_ds = valid_ds.prefetch(autotune)
        
        return train_ds, valid_ds


class LoggingManager:
    """Handles structured logging."""
    
    @staticmethod
    def log_training_info(dataset_config: DatasetConfig, training_config: TrainingConfig, 
                         network_config: NetworkConfig, train_ds: tf.data.Dataset):
        """Log comprehensive training information."""
        logging.info(f"[INFO] Training Start Time: {datetime.datetime.now()}")
        logging.info(f"[INFO] User defined dataset name: {dataset_config.name}")
        
        # Dataset info
        for batch_data in train_ds.take(1):
            x = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            _, h, w, c = x.shape
            logging.info(f"dataset one batch info: {x.shape}")
            logging.info(f"signal rescale to: ({x.numpy().min()}, {x.numpy().max()})")
            
            assert c == training_config.input_image_channel
            input_size = dataset_config.crop_size or training_config.input_image_size
            assert h == input_size
        
        # Preprocessing info
        logging.info("[INFO] Preprocessing Configuration:")
        logging.info(f"  - Crop Size: {dataset_config.crop_size if dataset_config.crop_size else 'same as image size'}")
        logging.info(f"  - Crop Type: {dataset_config.crop_type}")
        logging.info(f"  - Crop Position: {dataset_config.crop_position}")
        logging.info(f"  - Data Augmentation: {dataset_config.augment}")
        
        # Training parameters
        logging.info(f"[INFO] Forward Training Steps: {network_config.timesteps}")
        logging.info(f"[INFO] Noise Scheduler: {network_config.scheduler}")
        logging.info(f"[INFO] Learning Rate Type: {training_config.lr_type}")
        logging.info(f"[INFO] Learning Rate: {training_config.learning_rate}")
        logging.info(f"[INFO] Batch Size: {training_config.batch_size}")
        logging.info(f"[INFO] Predict Type: {training_config.pred_type}")
        logging.info(f"[INFO] Loss Function: {training_config.loss_fn}")
        logging.info(f"[INFO] Total Epochs: {training_config.epochs}")
        logging.info(f"[INFO] Steps per Epoch: {training_config.steps_per_epoch}")


# =====================
# Main Workflow Classes
# =====================

class DiffusionTrainer:
    """Handles the complete training workflow."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.dataset_config, self.training_config, self.network_config, self.imgen_config = \
            ConfigManager.parse_config(config_file)
    
    def train(self):
        """Execute the training workflow."""
        # Setup GPU
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        
        # Determine image size
        input_image_size = self.dataset_config.crop_size or self.training_config.input_image_size
        
        # Build models
        network, ema_network = ModelBuilder.build_models(
            input_image_size, self.training_config.input_image_channel, self.network_config
        )
        network_inputs = network.inputs
        network_outputs = network(network_inputs, training=False)
        network_encoded = keras.Model(
            inputs=network_inputs, outputs=network_outputs, name="DDPM_Network")
        # show inputs, outputs and total parameters only, no display of model graph details
        network_encoded.summary()
        
        # Create diffusion utilities and model
        diff_util_train = ModelBuilder.create_diffusion_utility(
            self.network_config, self.training_config, reverse_stride=1, clip_denoise=False
        )
        
        ddpm = DiffusionModel(
            network=network,
            ema_network=ema_network,
            diff_util=diff_util_train,
            num_classes=self.network_config.num_classes,
            save_period=self.training_config.save_period,
        )
        
        # Setup directories and logging
        logging_dir = DirectoryManager.setup_training_directories(
            self.dataset_config, self.training_config, self.network_config, self.config_file
        )
        
        # Load existing model if continuing training
        if not self.training_config.is_new_train and self.training_config.trained_h5:
            ddpm.ema_network.load_weights(self.training_config.trained_h5)
            ddpm.network.set_weights(ddpm.ema_network.get_weights())
        
        # Prepare datasets
        train_ds, valid_ds = DatasetManager.prepare_datasets(self.dataset_config, self.training_config)
        
        # Log training information
        t0 = time.time()
        LoggingManager.log_training_info(self.dataset_config, self.training_config, 
                                        self.network_config, train_ds)
        
        # Setup training components
        lr_schedule = ModelBuilder.create_lr_schedule(self.training_config)
        loss_fn = ModelBuilder.create_loss_function(self.training_config.loss_fn)
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)
        
        # Setup callbacks
        callbacks = [
            CSVLogger(os.path.join(logging_dir, "log.csv"), append=True, separator=","),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: ddpm.save_ema_model(epoch, savedir=logging_dir)
            ),
            keras.callbacks.LambdaCallback(
                on_train_end=lambda logs: None  # Placeholder for generate_images_and_save
            ),
            TQDMProgressBar(),
        ]

        if self.training_config.inline_gen_enable:
            callbacks.append(InlineImageGenerationCallback(
                period=self.training_config.inline_gen_period,
                num_images=self.training_config.inline_gen_nums,
                reverse_stride=self.training_config.inline_gen_reverse_stride,
                savedir=os.path.join(logging_dir, 'inline_gen'),
                labels=None,
            ))
        
        # Compile and train
        ddpm.compile(loss=loss_fn, optimizer=optimizer)
        ddpm.fit(
            train_ds,
            epochs=self.training_config.epochs,
            steps_per_epoch=self.training_config.steps_per_epoch,
            callbacks=callbacks,
        )
        
        # Log completion
        delta_time = np.around((time.time() - t0) / 3600.0, 4)
        logging.info(f"[INFO] Training End: {datetime.datetime.now()}, elapsed time: {delta_time} hours")
    

class ImageGenerator:
    """Handles the image generation workflow."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.dataset_config, self.training_config, self.network_config, self.imgen_config = \
            ConfigManager.parse_config(config_file)
    
    def generate(self):
        """Execute the image generation workflow."""
        # Validate generation config
        assert self.imgen_config.gen_task is not None
        assert self.imgen_config.model_path and os.path.isfile(self.imgen_config.model_path)
        
        # Setup generation directory
        model_dir = os.path.dirname(self.imgen_config.model_path)
        gen_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if self.imgen_config.gen_save_dir is None:
            gen_steps = str(self.network_config.timesteps // self.imgen_config.reverse_stride) + "steps"
            gen_save_dir = "_".join([
                "imgen", gen_steps, self.imgen_config.gen_task, os.uname().nodename, gen_date
            ])
            self.imgen_config.gen_save_dir = os.path.join(model_dir, gen_save_dir)
        
        os.makedirs(self.imgen_config.gen_save_dir, exist_ok=True)
        
        # Setup logging
        DirectoryManager.init_logging(os.path.join(self.imgen_config.gen_save_dir, "imgen.log"))
        self._log_generation_info()
        
        # Create diffusion utility for inference
        diff_util_infer = ModelBuilder.create_diffusion_utility(
            self.network_config, self.training_config,
            reverse_stride=self.imgen_config.reverse_stride,
            ddim_eta=self.imgen_config.ddim_eta,
            clip_denoise=self.imgen_config.clip_denoise
        )
        
        if self.imgen_config.new_image_size is not None:
            input_image_size = self.imgen_config.new_image_size
        else:
            input_image_size = self.dataset_config.crop_size or self.training_config.input_image_size
        
        # Build models
        _, ema_model = ModelBuilder.build_models(
            input_image_size, self.training_config.input_image_channel, self.network_config
        )
        # Show inputs, outputs and total parameters only, no display of model graph details
        ema_model_encoded = keras.Model(
            inputs=ema_model.inputs, outputs=ema_model(ema_model.inputs, training=False), name="DDPM_Network"
        )
        ema_model_encoded.summary()

        # Load model weights
        ema_model.load_weights(self.imgen_config.model_path)

        # Load model
        #ema_model = keras.models.load_model(
        #    self.imgen_config.model_path,
        #    custom_objects={"TimeEmbedding": TimeEmbedding}
        #)

        ddpm_infer = DiffusionModel(
            network=ema_model,
            ema_network=ema_model,
            diff_util=diff_util_infer,
            num_classes=self.network_config.num_classes,
        )
        
        # Setup generation parameters
        base_images, labels = self._prepare_generation_inputs()
        
        # Set random seed
        if self.imgen_config.random_seed:
            tf.random.set_seed(self.imgen_config.random_seed)
        
        # Generate images
        t0 = time.time()
        ddpm_infer.generate_images_and_save(
            gen_task=self.imgen_config.gen_task,
            reverse_stride=self.imgen_config.reverse_stride,
            savedir=self.imgen_config.gen_save_dir,
            num_images=self.imgen_config.num_gen_images,
            clip_denoise=self.imgen_config.clip_denoise,
            base_images=base_images,
            labels=labels,
            inpaint_mask=None,
            freeze_channel=self.imgen_config.freeze_channel,
            export_intermediate=self.imgen_config.export_interm,
            enable_memory_logging=True,
            memory_log_path=os.path.join(self.imgen_config.gen_save_dir, "memory_log.txt"),
            save_to_npz=True,
        )
        
        # Log completion
        delta_time = np.around((time.time() - t0), 1)
        logging.info(f"Generation images completed with {delta_time} seconds")
        logging.info(f"[IMGEN] {self.imgen_config.num_gen_images} images generated and saved to {self.imgen_config.gen_save_dir}")
    
    def _log_generation_info(self):
        """Log generation configuration."""
        logging.info(f"[IMGEN] Start to generate images using model: {self.imgen_config.model_path}")
        logging.info(f"[IMGEN] Generation Task: {self.imgen_config.gen_task}")
        logging.info(f"[IMGEN] External npz: {self.imgen_config.external_npz_input}")
        logging.info(f"[IMGEN] freeze channel: {self.imgen_config.freeze_channel}")
        logging.info(f"[IMGEN] class label: {self.imgen_config.class_label}")
        logging.info(f"[IMGEN] Model Predict Type: {self.training_config.pred_type}")
        logging.info(f"[IMGEN] DDIM eta = {self.imgen_config.ddim_eta}")
        logging.info(f"[IMGEN] Set Random Seed: {self.imgen_config.random_seed}")
        logging.info(f"[IMGEN] clip_denoise: {self.imgen_config.clip_denoise}")
        logging.info(f"[IMGEN] hostname: {os.uname().nodename}")
        logging.info(f"[IMGEN] TF version: {tf.__version__}")
    
    def _prepare_generation_inputs(self) -> Tuple[Optional[np.ndarray], Optional[tf.Tensor]]:
        """Prepare base images and labels for generation."""
        # Validate generation task
        if self.imgen_config.gen_task == "random_uncond":
            pass  # No additional validation needed
        elif self.imgen_config.gen_task == 'channel_inpaint':
            assert self.imgen_config.external_npz_input is not None
            assert self.imgen_config.freeze_channel is not None
        elif self.imgen_config.gen_task == 'class_cond':
            assert self.imgen_config.class_label is not None
            assert self.network_config.num_classes is not None
        else:
            raise NotImplementedError(f"Generation task {self.imgen_config.gen_task} not implemented")
        
        # Prepare base images
        base_images = None
        if self.imgen_config.external_npz_input:
            assert os.path.isfile(self.imgen_config.external_npz_input)
            base_images = np.load(self.imgen_config.external_npz_input)['images'].astype(np.float32)
            assert len(base_images.shape) == 4
            self.imgen_config.num_gen_images = base_images.shape[0]
            base_images = 2.0 * base_images - 1.0
        
        # Prepare labels
        labels = None
        if isinstance(self.imgen_config.class_label, int):
            labels = tf.fill([self.imgen_config.num_gen_images], int(self.imgen_config.class_label))
        elif isinstance(self.imgen_config.class_label, list):
            label_choices = np.random.choice(
                self.imgen_config.class_label * self.imgen_config.num_gen_images, 
                self.imgen_config.num_gen_images
            )
            labels = tf.constant(label_choices, tf.int32)
        
        return base_images, labels


# =====================
# Main Entry Point
# =====================

def main():
    """Main entry point for training or image generation."""
    parser = argparse.ArgumentParser(description="DDPM v3 Training and Image Generation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--training", action='store_true', help="Run training mode")
    parser.add_argument("--imgen", action='store_true', help="Run image generation mode")
    parser.add_argument("--enable_xla", action='store_true', help='Enable XLA JIT compilation')
    
    args = parser.parse_args()
    
    # Enable XLA if requested
    if args.enable_xla:
        tf.config.optimizer.set_jit(True)
    
    # Execute requested mode
    if args.training:
        trainer = DiffusionTrainer(args.config)
        trainer.train()
    elif args.imgen:
        generator = ImageGenerator(args.config)
        generator.generate()
    else:
        print("No action specified. Use --training or --imgen.")


if __name__ == "__main__":
    main()
