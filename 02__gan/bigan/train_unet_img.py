from __future__ import print_function, division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from data_loader import DataLoader
from utils import build_model
from utils_img import get_loss_fn, Monitor

class MLModel(Model):
    def __init__(self,
        in_channels,
        out_channels,
        generator_structure_dict,
        discriminator_structure_dict=None
    ):
        super().__init__()

        # Define loss trackers.
        self.g_loss_tracker = metrics.Mean(name="g_loss")
        self.val_g_loss_tracker = metrics.Mean(name="g_loss")

        # Build generator.
        self.generator, self.generator_ambit = build_model(
            in_channels=in_channels, out_channels=out_channels, 
            kernels=generator_structure_dict["KERNEL_SIZE"],
            channels=generator_structure_dict["CHANNELS"],
            paddings=generator_structure_dict["PADDING"],
            symmetries=generator_structure_dict["SYMMETRY"],
            activations=generator_structure_dict["ACTIVATION_FUNCTIONS"],
            batchNorms=generator_structure_dict["BATCH_NORMALIZATION"],
            cba=generator_structure_dict["_BATCH_NORMALIZATION"]["CBA"],
            skipConnections=generator_structure_dict["SKIP_INPUT"],
            skipType=generator_structure_dict["_SKIP_TYPE"],
            dilations=generator_structure_dict["DILATION_RATE"],
            reg_l1=generator_structure_dict["REG_L1"],
            reg_l2=generator_structure_dict["REG_L2"],
            target=False
        )

        self.cor_ambit = self.generator_ambit // 2
        print(f"generator_ambit: {self.generator_ambit}")

    @property
    def metrics(self):
        return [
            self.g_loss_tracker,
            self.val_g_loss_tracker
        ]
    
    def compile(self, g_optimizer, g_loss_fn):
        """ Override compile of tf.keras.models.Model."""
        super().compile()
        self.g_optimizer = g_optimizer
        self.g_loss_fn = g_loss_fn
    
    def g_loss(self, fake_image, real_image):
        total_loss = self.g_loss_fn(fake_image, real_image)
        return total_loss

    @tf.function
    def train_step(self, data):
        input_image, real_image = data
        with tf.GradientTape() as g_tape:
            fake_image = self.generator(input_image, training=True)
            g_loss_value = self.g_loss(fake_image, real_image)

        g_gradients = g_tape.gradient(g_loss_value, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        # Monitor loss.
        self.g_loss_tracker.update_state(g_loss_value)

        return {
            "g_loss": self.g_loss_tracker.result()
        }
    
    @tf.function
    def test_step(self, data):
        input_image, real_image = data
        fake_image = self.generator(input_image, training=False)
        g_loss_value = self.g_loss(fake_image, real_image)

        # Monitor loss.
        self.val_g_loss_tracker.update_state(g_loss_value)

        return {
            "g_loss": self.val_g_loss_tracker.result()
        }
    
    def call(self, inputs):
        return self.generator(inputs, training=False)


class Trainer():
    def __init__(self,
        model,
        logging_path,
        dataset_config,
        target_image_ambit,
        generator_hyperparameters,
        discriminator_hyperparameters=None
    ):
        self.model = model
        self.logging_path = logging_path
        self.target_image_ambit = target_image_ambit
        self.target_output_size = target_image_ambit * 2 + 1
        self.crop_size = model.generator_ambit + self.target_output_size
        self.cor_ambit = model.generator_ambit // 2
        print(f"crop_size: {self.crop_size}")
        # dataset_config
        self.data_dir = dataset_config["DATA_DIR"]
        self.asd_file = dataset_config["ASD_FILE"]
        self.dataset_config = dataset_config["DATASET_CONFIG"]
        self.data_name = dataset_config["NAME"]
        self.tanh_norm = dataset_config["TANH_NORM"]
        self.img_idx = dataset_config["IMG_IDX"]
        self.crop_type = dataset_config["CROP_TYPE"]
        self.in_channels = dataset_config["IN_CHANNELS"]
        self.out_channels = dataset_config["OUT_CHANNELS"]
        self.trainset_num = dataset_config["TRAINSET_NUM"]
        self.valset1_num = self.valset2_num = dataset_config["VALSET_NUM"]
        self.data_standardization = dataset_config["DATA_STANDARDIZATION"]

        self.trainset_path = [self.data_dir, self.data_name, "train"]
        self.valset_1_path = [self.data_dir, self.data_name, "val"]
        self.valset_2_path = [self.data_dir, self.data_name, "train"]

        self.train_loader = DataLoader(
            dataset_path=self.trainset_path,
            asd_file=self.asd_file,
            dataset_config=self.dataset_config,
            datasetnum=self.trainset_num,
            tanh_norm=self.tanh_norm,
            crop_size=self.crop_size,
            crop_type=self.crop_type,
            cor_ambit=self.cor_ambit,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )
        self.val_loader_1 = DataLoader(
            dataset_path=self.valset_1_path,
            asd_file=self.asd_file,
            dataset_config=self.dataset_config,
            datasetnum=self.valset1_num,
            tanh_norm=self.tanh_norm,
            crop_size=self.crop_size,
            crop_type="center",
            cor_ambit=self.cor_ambit,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )
        self.val_loader_2 = DataLoader(
            dataset_path=self.valset_2_path,
            asd_file=self.asd_file,
            dataset_config=self.dataset_config,
            datasetnum=self.valset2_num,
            tanh_norm=self.tanh_norm,
            crop_size=self.crop_size,
            crop_type="center",
            cor_ambit=self.cor_ambit,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

        # Define lr scheduler and optimizer.
        g_optimizer = Adam(learning_rate=generator_hyperparameters["LEARNING_RATE"], beta_1=0.6)

        g_loss_fn = get_loss_fn(generator_hyperparameters["LOSS"])
        print("LOSS_FN : ", g_loss_fn)

        self.start_epoch = 0

        # Compile model.
        self.model.compile(
            g_loss_fn=g_loss_fn, 
            g_optimizer=g_optimizer
        )

    def train(self,
            training_hyperparameters
        ):

        epochs = training_hyperparameters["EPOCHS"]
        val_epoch = training_hyperparameters["VAL_EPOCH"]
        iterations = training_hyperparameters["ITERATIONS"]
        val_iter = training_hyperparameters["VAL_ITER"]
        batch_size = training_hyperparameters["BATCH_SIZE"]
        assert not (iterations > 0 and val_iter > 0), "Only support fit(epochs)"
        assert (epochs > 0 and val_epoch > 0)

        """ Train model with trainset."""
        trainset = self.train_loader.load_dataset(
            batch_size=batch_size,
            img_idx=self.img_idx,
            is_train=True,
        )
        valset_1 = self.val_loader_1.load_dataset(
            batch_size=1,
            img_idx=self.img_idx,
        )
        valset_2 = self.val_loader_2.load_dataset(
            batch_size=1,
            img_idx=self.img_idx,
        )
        print("trainset_size:", trainset.cardinality().numpy())
        print("valset_1_size:", valset_1.cardinality().numpy())
        print("valset_2_size:", valset_2.cardinality().numpy())

        csv_logger = CSVLogger(self.logging_path+'/log.csv', append=True, separator=',')
        callbacks = [
            Monitor(
                valset_1=valset_1,
                valset_2=valset_2,
                logging_path=self.logging_path,
                target_image_ambit = self.target_image_ambit,
                generator_ambit = self.model.generator_ambit,
                tanh_norm = self.tanh_norm,
                val_epoch = val_epoch
            ),
            csv_logger
        ]

        # self.profile=True
        # if self.profile:
        #     callbacks.append(
        #         tf.keras.callbacks.TensorBoard(
        #             log_dir=self.logging_path, profile_batch="10, 15"
        #         )
        #     )

        # Train model with trainset.
        self.model.fit(
            trainset,
            epochs=epochs,
            initial_epoch=self.start_epoch,
            validation_data=valset_1,
            callbacks=callbacks
        )

        print(self.model.summary())
