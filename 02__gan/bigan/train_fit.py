import os, sys, argparse, shutil, logging
import importlib.util
import tensorflow as tf
from pathlib import Path
from utils import read_config, init_logging, transfer_weight


def run_img_training():
    generator_dict = image_training_dict["GENERATOR"]
    discriminator_dict = image_training_dict["DISCRIMINATOR"]

    # Define model.
    model = img_training.MLModel(
        in_channels=dataset_dict["IN_CHANNELS"],
        out_channels=dataset_dict["OUT_CHANNELS"],
        generator_structure_dict=generator_dict["STRUCTURE"],
        discriminator_structure_dict=None if discriminator_dict==False else discriminator_dict["STRUCTURE"]
    )

    if image_training_dict["ENABLE"]:
        logging.info("\n\n- Image Training Started\n\n")

        result_path = f"{logging_path}/image_training"
        Path(result_path).resolve().mkdir(parents=True, exist_ok=True)

        # Define trainer.
        trainer = img_training.Trainer(
            model,
            logging_path=result_path,
            dataset_config=dataset_dict,
            target_image_ambit=image_training_dict["TARGET_IMAGE_AMBIT"],
            generator_hyperparameters=generator_dict["HYPER_PARAMETERS"],
            discriminator_hyperparameters=None if discriminator_dict==False else discriminator_dict["HYPER_PARAMETERS"]
        )
        
        trainer.train(
            training_hyperparameters=image_training_dict["HYPER_PARAMETERS"]
        )

        logging.info(f"Image Training Results Dir: {result_path}")
        logging.info("\n\n- Image Training Finished\n\n")
        
        return result_path, model
    
    return None, model

def run_epe_training():
    result_path = f"{logging_path}/epe_training"
    Path(result_path).resolve().mkdir(parents=True, exist_ok=True)

    generator_dict = epe_training_dict["GENERATOR"]
    discriminator_dict = epe_training_dict["DISCRIMINATOR"]

    # Define model.
    model = epe_training.MLModel(
        in_channels=dataset_dict["IN_CHANNELS"],
        out_channels=dataset_dict["OUT_CHANNELS"],
        generator_structure_dict=generator_dict["STRUCTURE"],
        discriminator_structure_dict=None if discriminator_dict==False else discriminator_dict["STRUCTURE"],
        training_hyperparameters=epe_training_dict["HYPER_PARAMETERS"]
    )

    # Load image trianing results
    if (image_training_dict["DISCRIMINATOR"] == False) and (epe_training_dict["DISCRIMINATOR"] == False):
        if img_training_weights == False: model.load_weights(f"{img_training_path}/G-ep_max")
        else: model.load_weights(img_training_weights)
        logging.info("-CNN weights loaded")
    else:
        if img_training_weights == False:
            img_training_model.load_weights(f"{img_training_path}/G-ep_max")
            logging.info(f"-Image training weights: {img_training_path}/G-ep_max")
        else:
            img_training_model.load_weights(img_training_weights)
            logging.info(f"-Image training weights: {img_training_weights}")
        # Transfer weights
        transfer_weight(img_training_model.generator, model.generator)
        logging.info("-Image training weights LOADED")
    #


    # Define trainer.
    trainer = epe_training.Trainer(
        model,
        logging_path=result_path,
        dataset_config=dataset_dict,
        target_image_ambit=epe_training_dict["TARGET_IMAGE_AMBIT"],
        generator_hyperparameters=generator_dict["HYPER_PARAMETERS"],
        discriminator_hyperparameters=None if discriminator_dict==False else discriminator_dict["HYPER_PARAMETERS"],
        img_training_target_ambit=image_training_dict["TARGET_IMAGE_AMBIT"]
    )

    trainer.train(
        training_hyperparameters=epe_training_dict["HYPER_PARAMETERS"]
    )

    logging.info(f"EPE Training Results Dir: {result_path}")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)

    parser = argparse.ArgumentParser(description="ML example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="unet")
    args = parser.parse_args()

    config_dict = read_config(args.config)

    # Set logging.
    logging_path = f"results/{args.run_name}"
    Path(logging_path).resolve().mkdir(parents=True, exist_ok=True)
    init_logging(f"{logging_path}/train.log")
    shutil.copy(args.config, f"{logging_path}/training_config.yaml")

    # DATASET
    dataset_dict = config_dict["DATASET"]

    # IMAGE TRAINING
    image_training_dict = config_dict["TRAINING"]["IMAGE_TRAINING"]
    # if image_training_dict["ENABLE"]:
    if image_training_dict["DISCRIMINATOR"] == False:
        img_training_script = "train_unet_img.py"
    else:
        img_training_script = "train_gan_img.py"
    #
    img_training_spec = importlib.util.spec_from_file_location(
        "module.name", img_training_script)
    img_training = importlib.util.module_from_spec(img_training_spec)
    sys.modules["module.name"] = img_training
    img_training_spec.loader.exec_module(img_training)

    img_training_path, img_training_model = run_img_training()

    # EPE TRAINING
    epe_training_dict = config_dict["TRAINING"]["EPE_TRAINING"]
    if epe_training_dict["ENABLE"]:
        logging.info("\n\n- EPE Training Started\n\n")

        if epe_training_dict["DISCRIMINATOR"] == False:
            epe_training_script = "train_unet_epe.py"
        else:
            epe_training_script = "train_gan_epe.py"
        
        img_training_weights = \
            epe_training_dict["INITIALIZATION"]["IMAGE_TRAINING_WEIGHTS"]
        if (img_training_weights == False) and (img_training_path == None):
            logging.info("-ERROR [Run image training before epe training]")
            sys.exit(1)
    
        epe_training_spec = importlib.util.spec_from_file_location(
            "module.name", epe_training_script)
        epe_training = importlib.util.module_from_spec(epe_training_spec)
        sys.modules["module.name"] = epe_training
        epe_training_spec.loader.exec_module(epe_training)

        run_epe_training()

        logging.info("\n\n- EPE Training Finished\n\n")
