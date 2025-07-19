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
import yaml
import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tqdm import tqdm

from diffusion_utils import DiffusionUtility
from layers import (
    kernel_init,
    TimeEmbedding,
    TimeMLP,
    ResidualBlock,
    DownSample,
    UpSample,
)
from unet import build_model
from diffusion_model import DiffusionModel
from data_loader import DataLoader
from callbacks import WarmUpCosine, TQDMProgressBar, InlineImageGenerationCallback


# =====================
# Utility Functions
# =====================

def init_logging(filename, checkpoint=None):
    """
    Initialize logging to file and console.
    """
    mode = "w+"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(filename, mode=mode), logging.StreamHandler()],
    )

def parse_config(config_path):
    """
    Parse YAML config file.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['DATASET'], cfg['TRAINING'], cfg['IMAGE_GENERATION']

def build_models(image_size, image_channel, widths, has_attention,
                 num_heads, num_res_blocks, norm_groups, block_size, temb_dim,
                 dropout_rate=0.0, kernel_size=3, use_cross_attention=False,
                 num_classes=None, class_emb_dim=None):
    """
    Builds the main and EMA models.
    """
    kwargs = dict(
        image_size=image_size,
        image_channel=image_channel,
        widths=widths,
        has_attention=has_attention,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        actf=keras.activations.swish,
        block_size=block_size,
        temb_dim=temb_dim,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        use_cross_attention=use_cross_attention,
        num_classes=num_classes,
        class_emb_dim=class_emb_dim,
    )
    network = build_model(**kwargs)
    ema_network = build_model(**kwargs)
    ema_network.set_weights(network.get_weights())
    return network, ema_network

def prepare_datasets(dataset_path, img_size, batch_size, crop_size=None,
                     label_key=None, dtype=tf.float32):
    """
    Prepare training and validation datasets with efficient prefetching.
    """
    autotune = tf.data.AUTOTUNE
    if os.path.isdir(dataset_path):
        dataloader = DataLoader(
            data_dir=dataset_path,
            img_size=img_size,
            crop_size=crop_size,
            label_key=label_key,
        )
        train_ds, valid_ds = dataloader._get_dataset()
    else:
        data = np.load(dataset_path)
        all_images = 2 * data['images'] - 1.0
        labels = data[label_key] if (label_key is not None and label_key in data) else None
        idx = np.arange(len(all_images))
        np.random.shuffle(idx)
        num_val = int(0.1 * len(all_images))
        if labels is None:
            train_ds = tf.data.Dataset.from_tensor_slices(all_images)
            valid_ds = tf.data.Dataset.from_tensor_slices(all_images[0:num_val])
            train_ds = train_ds.map(lambda x: tf.cast(x, dtype), num_parallel_calls=autotune)
            valid_ds = valid_ds.map(lambda x: tf.cast(x, dtype), num_parallel_calls=autotune)
            train_ds = train_ds.cache().shuffle(buffer_size=10000).repeat()
            valid_ds = valid_ds.cache()
        else:
            train_ds = tf.data.Dataset.from_tensor_slices((all_images, labels))
            valid_ds = tf.data.Dataset.from_tensor_slices((all_images[0:num_val], labels[0:num_val]))
            train_ds = train_ds.map(lambda x,y: (tf.cast(x, dtype), tf.cast(y, tf.int32)), num_parallel_calls=autotune)
            valid_ds = valid_ds.map(lambda x,y: (tf.cast(x, dtype), tf.cast(y, tf.int32)), num_parallel_calls=autotune)
            train_ds = train_ds.cache().shuffle(buffer_size=10000).repeat()
            valid_ds = valid_ds.cache()

    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    valid_ds = valid_ds.batch(batch_size)
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    return train_ds, valid_ds

# =====================
# Main Logic Functions
# =====================

def train_model(config_file):
    """
    Handles the training workflow.
    """
    dataset_dict, training_dict, imgen_dict = parse_config(config_file)
    # dataset
    dataset_name   = dataset_dict['NAME']
    dataset_path   = dataset_dict['PATH']
    label_key      = dataset_dict.get('LABEL_KEY')
    crop_size      = dataset_dict['PREPROCESSING'].get('CROP_SIZE')
    crop_type      = dataset_dict['PREPROCESSING'].get('CROP_TYPE')

    # input shape
    input_image_size    = training_dict['INPUT_IMAGE_SIZE']
    input_image_channel = training_dict['INPUT_IMAGE_CHANNEL']
    # training
    training_output_dir = training_dict['OUTPUT_DIR']
    os.makedirs(training_output_dir, exist_ok=True)
    is_new_train       = training_dict['IS_NEW_TRAIN']
    trained_h5         = training_dict['TRAINED_H5']
    pred_type          = training_dict['PRED_TYPE']
    loss_fn            = training_dict['LOSS_FN']
    # control the inline image gen during training
    inline_gen_dict    = training_dict['INLINE_GEN']
    inline_gen_enable  = inline_gen_dict['ENABLE']
    inline_gen_nums    = inline_gen_dict.get('NUMS', 20)
    inline_gen_period  = inline_gen_dict.get('PERIOD', 10)
    inline_gen_reverse_stride = inline_gen_dict.get('REVERSE_STRIDE', 10)
    # network
    scheduler          = training_dict['NETWORK']['SCHEDULER']
    timesteps          = training_dict['NETWORK']['TIMESTEPS']
    num_res_blocks     = training_dict['NETWORK']['NUM_RES_BLOCKS']
    block_size         = training_dict['NETWORK']['BLOCK_SIZE']
    norm_groups        = training_dict['NETWORK']['NORM_GROUPS']
    first_channel      = training_dict['NETWORK']['FIRST_CHANNEL']
    channel_multiplier = training_dict['NETWORK']['CHANNEL_MULTIPLIER']
    has_attention      = training_dict['NETWORK']['HAS_ATTENTION']
    num_heads          = training_dict['NETWORK']['NUM_HEADS']
    assert len(channel_multiplier)==len(has_attention)
    widths = [first_channel * mult for mult in channel_multiplier]
    temb_dim           = training_dict['NETWORK']['TIME_EMB_DIM']
    dropout_rate       = training_dict['NETWORK'].get('DROPOUT_RATE', 0.0)
    kernel_size        = training_dict['NETWORK'].get('KERNEL_SIZE', 3)
    use_cross_attention = training_dict['NETWORK'].get('USE_CROSS_ATTENTION', False)
    num_classes       = training_dict['NETWORK'].get('NUM_CLASSES')
    class_emb_dim     = training_dict['NETWORK'].get('CLASS_EMB_DIM')

    # hyper-parameters
    epochs          = training_dict['HYPER_PARAMETERS']['EPOCHS']
    batch_size      = training_dict['HYPER_PARAMETERS']['BATCH_SIZE']
    steps_per_epoch = training_dict['HYPER_PARAMETERS']['STEPS_PER_EPOCH']
    lr_type         = training_dict['HYPER_PARAMETERS']['LR_TYPE']
    learning_rate   = training_dict['HYPER_PARAMETERS']['LEARNING_RATE']
    warmup_steps    = training_dict['HYPER_PARAMETERS']['WARMUP_STEPS']
    total_steps = None if steps_per_epoch is None else epochs * steps_per_epoch
    
    # GPU devices
    gpus = tf.config.list_physical_devices("GPU")
    if gpus: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

    # Build (initialize) the unet model
    network, ema_network = build_models(
        image_size=input_image_size,
        image_channel=input_image_channel,
        widths=widths,
        has_attention=has_attention,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        block_size=block_size,
        temb_dim=temb_dim,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
        use_cross_attention=use_cross_attention,
        num_classes=num_classes,
        class_emb_dim=class_emb_dim,
    )

    network.summary()

    # Get an instance of the Gaussian Diffusion utilities
    # util for training
    diff_util_train = DiffusionUtility(
        b0=0.1, b1=20, timesteps=timesteps, 
        scheduler=scheduler, pred_type=pred_type, reverse_stride=1, 
        clip_denoise=False,
        )

    # Get the diffusion model (keras.Model)
    ddpm = DiffusionModel(
        network=network,
        ema_network=ema_network,
        diff_util=diff_util_train,
        num_classes=num_classes,
        )
        
    assert dataset_name is not None
        
    (train_ds, valid_ds) = (None, None)
    input_shape = (input_image_size, input_image_size, input_image_channel)
    # create folder for dataset tag
    dataset_tag = os.path.join(
        os.path.abspath(training_output_dir), 
        "{}_{}x{}x{}".format(dataset_name, *input_shape),
        )
    os.makedirs(dataset_tag, exist_ok=True)
    # create model nametag
    model_nametag = "unet"+str(first_channel)+"m"+"".join(map(str, channel_multiplier))
    model_nametag = model_nametag+"g"+str(norm_groups)
    model_nametag = model_nametag+"rb"+str(num_res_blocks)
    model_nametag = model_nametag+"bk"+str(block_size)
    tr_output_dir = os.path.join(dataset_tag, model_nametag)
    if not os.path.isdir(tr_output_dir): os.mkdir(tr_output_dir)

    dateID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dateID = "_".join([scheduler,str(timesteps),pred_type, loss_fn, dateID])
        
    if is_new_train:
        logging_dir = os.path.join(tr_output_dir, dateID)
        os.mkdir(logging_dir)
        init_logging(os.path.join(logging_dir, "train.log"))
        logging.info("[INFO] Start a new training")
    else: 
        if trained_h5 is None:
            logging.info('if "IS_NEW_TRAIN: false", you must provide a trained .h5 file')
            return 
        else:
            # load .h5 file
            trained_h5 = os.path.abspath(trained_h5)
            restored_model_ID = os.path.basename(os.path.dirname(trained_h5))
            restored_model_ID = restored_model_ID.split("_")[-1]
            ddpm.ema_network.load_weights(trained_h5)
            ddpm.network.set_weights(ddpm.ema_network.get_weights())

        logging_dir = os.path.join(tr_output_dir, dateID+"_from_"+restored_model_ID)
        os.mkdir(logging_dir)
        init_logging(os.path.join(logging_dir,"train.log"))
        logging.info("[INFO] Restoring model from: {}".format(trained_h5))
        logging.info("[INFO] Continuous Transfer training ...")
        
    shutil.copy(config_file, os.path.join(logging_dir, "training_config.yaml"))
    # end of creating logging_dir 

    logging.info("[INFO] Training Start Time: {}".format(datetime.datetime.now()))
    t0 = time.time()
        
    assert dataset_path is not None
    logging.info("[INFO] User defined dataset name:{}".format(dataset_name))
    train_ds, valid_ds = prepare_datasets(
        dataset_path,
        img_size=input_image_size,
        batch_size=batch_size,
        crop_size=crop_size,
        label_key=label_key,
        )
          
    # get input image shape
    for _batch_data in train_ds.take(1):
        if isinstance(_batch_data, (list, tuple)):
            x = _batch_data[0]
        else:
            x = _batch_data
        _, h, w, c = x.shape
        logging.info("dataset one batch info: {}".format(x.shape))
        logging.info("signal rescale to: ({},{})".format(x.numpy().min(), x.numpy().max()))
    assert c==input_image_channel
    assert h==input_image_size
    
    logging.info("[INFO] Forward Training Steps: {}".format(timesteps))
    logging.info("[INFO] Noise Scheduler: {} ".format(scheduler))
    logging.info("[INFO] Learning Rate Type: {}".format(lr_type))
    logging.info("[INFO] Learning Rate: {}".format(learning_rate))
    logging.info("[INFO] Batch Size: {}".format(batch_size))
    logging.info("[INFO] Predict Type: {}".format(pred_type))
    logging.info("[INFO] Loss Function: {}".format(loss_fn))
    logging.info("[INFO] Total Epochs: {}".format(epochs))
    logging.info("[INFO] Steps per Epoch: {}".format(steps_per_epoch))

    csv_logger = CSVLogger(os.path.join(logging_dir, "log.csv"), append=True, separator=",")
    #best_ckpt_path = os.path.join(logging_dir, "dm_best.weights.h5")
    #callback_save_ema_best = keras.callbacks.ModelCheckpoint(
    #  filepath=best_ckpt_path,
    #  save_weights_only=True,
    #  monitor='val_loss',
    #  mode='min',
    #  save_best_only=True,
    #  )
    
    save_ema_models_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: ddpm.save_ema_model(epoch, savedir=logging_dir))
    train_end_imgen_callback = keras.callbacks.LambdaCallback(
        on_train_end=lambda logs: ddpm.generate_images_and_save(logs=None))
    
    if inline_gen_enable:
        inline_imgen_callback = InlineImageGenerationCallback(
            period=inline_gen_period,
            num_images=inline_gen_nums,
            reverse_stride=inline_gen_reverse_stride,
            savedir = os.path.join(logging_dir, 'inline_gen'),
            labels=None,
        )
    
    callback_list = [
        csv_logger,
        save_ema_models_callback,
        train_end_imgen_callback,
        TQDMProgressBar(),
        #InlineEvalCallback(valid_ds, eval_interval=10000, savedir=logging_dir),
        #keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
        #                              restore_best_weights=True),
        ]
    if inline_gen_enable:
        callback_list.append(inline_imgen_callback)

    if loss_fn == "MAE":
        loss_fn = keras.losses.MeanAbsoluteError()
    elif loss_fn == 'MSE':
        loss_fn = keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError
        

    if lr_type == 'constant':
        lr_schedule = learning_rate

    elif lr_type == 'warmup_cosine':
        assert warmup_steps is not None
        lr_schedule = WarmUpCosine(
            base_lr = learning_rate,
            warmup_steps = warmup_steps,
            total_steps = total_steps,
            )

    elif lr_type == 'cosine_decay':
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = learning_rate,
            decay_steps = 10000,
            alpha = 0.0,
            )
    else:
        raise NotImplementedError

    optimizer = keras.optimizers.AdamW(
        learning_rate = lr_schedule,
        #weight_decay=1.0e-5,
        )

    # Compile the model
    ddpm.compile(loss=loss_fn,optimizer=optimizer)

    # Train the model
    ddpm.fit(
        train_ds,
        #validation_data=valid_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callback_list,
        )
    
    deltaT = np.around((time.time() - t0)/3600.0, 4)
    nowT = datetime.datetime.now()
    logging.info("[INFO] Training End: {}, elapsed time: {} hours".format(nowT, deltaT))
    
    return None

def generate_images(config_file):
    """
    Handles the image generation workflow.
    """
    dataset_dict, training_dict, imgen_dict = parse_config(config_file)
    # for image generation (imgen)
    imgen_model_path = imgen_dict['MODEL_PATH']
    num_gen_images   = imgen_dict['NUM_GEN_IMAGES']
    export_interm    = imgen_dict['EXPORT_INTERM']
    reverse_stride   = imgen_dict['REVERSE_STRIDE']
    gen_inputs       = imgen_dict['GEN_INPUTS']
    freeze_channel   = imgen_dict.get('FREEZE_CHANNEL', 0)
    gen_output_dir   = imgen_dict['GEN_OUTPUT_DIR']
    ddim_eta         = imgen_dict['DDIM_ETA']
    random_seed      = imgen_dict['RANDOM_SEED']
    class_label      = imgen_dict.get('CLASS_LABEL')
    clip_denoise     = imgen_dict.get('CLIP_DENOISE', False)
    num_classes      = training_dict['NETWORK'].get('NUM_CLASSES')
    timesteps        = training_dict['NETWORK']['TIMESTEPS']
    scheduler        = training_dict['NETWORK']['SCHEDULER']
    pred_type        = training_dict['PRED_TYPE']

    if gen_inputs is not None:
        gen_inputs = np.load(gen_inputs)['images']
        gen_inputs = 2.0*gen_inputs -1 
        print(gen_inputs.shape)

    assert imgen_model_path is not None
    assert os.path.isfile(imgen_model_path)
    model_dir = os.path.dirname(imgen_model_path)
    gen_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # util for inference
    diff_util_infer = DiffusionUtility(
        b0=0.1, b1=20, timesteps=timesteps, 
        scheduler=scheduler, pred_type=pred_type, reverse_stride=reverse_stride,
        ddim_eta=ddim_eta, clip_denoise=clip_denoise,
        )
    
    ema_model = keras.models.load_model(
        imgen_model_path,
        custom_objects={
            "TimeEmbedding": TimeEmbedding,
        },
    )
    
    ddpm_infer = DiffusionModel(
        network=ema_model,
        ema_network=ema_model,
        diff_util=diff_util_infer,
        num_classes=num_classes,
        )
    
    if gen_output_dir is None: 
        gen_steps = str(diff_util_infer.timesteps // diff_util_infer.reverse_stride)+"steps"
        gen_dir ="_".join([
            "imgen", gen_steps, "tf"+tf.__version__, os.uname().nodename, gen_date])
        gen_dir = os.path.join(model_dir, gen_dir)
        os.mkdir(gen_dir)
    else:
        gen_dir = gen_output_dir
        os.makedirs(gen_dir, exist_ok=True)
        
    init_logging(os.path.join(gen_dir, "imgen.log"))
    logging.info("[IMGEN] Start to generate images using model: {}".format(imgen_model_path))
    logging.info("[IMGEN] Model Predict type: {}".format(diff_util_infer.pred_type))
    logging.info("[IMGEN] DDIM eta = {}".format(diff_util_infer.ddim_eta))
    logging.info("[IMGEN] Set random seed: {}".format(random_seed))
    logging.info("[IMGEN] clip_denoise: {}".format(clip_denoise))
    logging.info("[IMGEN] hostname: {}".format(os.uname().nodename))
    logging.info("[IMGEN] TF version: {}".format(tf.__version__))

    tf.random.set_seed(random_seed)

    labels = None
    if isinstance(class_label, int):
        labels = tf.fill([num_gen_images], int(class_label))
    elif isinstance(class_label, list):
        labels = np.random.choice(class_label*num_gen_images, num_gen_images)
        labesl = tf.constant(labels, tf.int32)
    else:
        labels = None

    t0 = time.time()

    ddpm_infer.generate_images_and_save(
        savedir=gen_dir,
        num_images=num_gen_images,
        clip_denoise=clip_denoise,
        gen_inputs=gen_inputs,
        labels=labels,
        inpaint_mask=None,
        freeze_channel=freeze_channel,
        export_intermediate=export_interm,
        enable_memory_logging=True,
        memory_log_path=os.path.join(gen_dir, "memory_log.txt"),
        )
    #
    deltaT = np.around((time.time()-t0), 1)
    logging.info("Generated {} images with {} seconds".format(num_gen_images, deltaT))
    logging.info("[IMGEN] Generated images saved to: {}".format(gen_dir))
    
    return None


def main():
    """
    Main entry point for training or image generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--training", dest='training', action='store_true')
    parser.add_argument("--imgen", dest='imgen', action='store_true')
    #parser.add_argument("--mixed_precision", action='store_true', help='Enable mixed float16 training')
    parser.add_argument("--enable_xla", action='store_true', help='Enable XLA JIT compilation')
    FLAGS, _ = parser.parse_known_args()

    #if FLAGS.mixed_precision:
    #    from tensorflow.keras import mixed_precision
    #    mixed_precision.set_global_policy("mixed_float16")
    
    if FLAGS.enable_xla:
        tf.config.optimizer.set_jit(True)

    #dtype = tf.float16 if FLAGS.mixed_precision else tf.float32
    #dataset_dict, training_dict, imgen_dict = parse_config(FLAGS.config)

    if FLAGS.training:
        train_model(FLAGS.config)
    elif FLAGS.imgen:
        generate_images(FLAGS.config)
    else:
        print("No action specified. Use --training or --imgen.")

if __name__ == "__main__":
    main()
