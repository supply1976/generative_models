import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from functools import partial
import random
import pandas as pd

import tensorflow as tf

np.set_printoptions(threshold=sys.maxsize)
input_has_context = "false"

def crop_images(img_A, img_B, img_name, crop_size, crop_type, cor_ambit, in_channels, out_channels, is_test):
    if input_has_context == "false":
        img_A = np.pad(img_A, ((cor_ambit, cor_ambit),(cor_ambit, cor_ambit),(0, 0)), 'constant', constant_values=-1)
        img_B = np.pad(img_B, ((cor_ambit, cor_ambit),(cor_ambit, cor_ambit),(0, 0)), 'constant', constant_values=-1)
    img_size = img_A.shape[0]
    if crop_type == "random":
        begin_x, begin_y = np.random.randint(cor_ambit, img_size - crop_size + 1 - cor_ambit, 2)
        end_x, end_y = begin_x + crop_size, begin_y + crop_size
        img_A = img_A[begin_x:end_x, begin_y:end_y, :]
        img_B = img_B[begin_x+cor_ambit:end_x-cor_ambit, begin_y+cor_ambit:end_y-cor_ambit, :]
    elif crop_type == "center":
        if crop_size != img_size:
            padsize = (img_size - crop_size) // 2
            begin = padsize
            end = -padsize
            img_A = img_A[begin:end, begin:end, :]
            img_B = img_B[begin+cor_ambit:end-cor_ambit, begin+cor_ambit:end-cor_ambit, :]
        elif cor_ambit>0:
            img_B = img_B[0+cor_ambit:img_size-cor_ambit, 0+cor_ambit:img_size-cor_ambit, :]
    else:
        raise ValueError(f"crop_type {crop_type} not recognized.")
    if is_test:
        return img_A, img_B, img_name
    else:
        return img_A, img_B


class DataLoader:
    def __init__(self, dataset_path, asd_file, dataset_config, datasetnum, tanh_norm, crop_size, crop_type, cor_ambit, in_channels, out_channels, data_standardization=False, merge_output_channels=False, is_test=False, subsampling=True):
        self.dataset_path = dataset_path
        self.tanh_norm = tanh_norm
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.cor_ambit = cor_ambit
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_output_channels = merge_output_channels
        self.is_test = is_test
        self.datasetnum = datasetnum

        #
        self.data_standardization = data_standardization

        crop_images_fn = crop_images
        if is_test:
            self.random_process = partial(crop_images_fn, crop_size=self.crop_size,
                    crop_type=self.crop_type, cor_ambit=self.cor_ambit,
                    in_channels=self.in_channels, out_channels=self.out_channels, is_test=is_test)
        else:
            self.random_process = partial(crop_images_fn, img_name="NA", crop_size=self.crop_size,
                    crop_type=self.crop_type, cor_ambit=self.cor_ambit,
                    in_channels=self.in_channels, out_channels=self.out_channels, is_test=is_test)

        #
        [dataset_dir, dataset_name, dataset_type] = self.dataset_path
        # dataset_config = f"{dataset_dir}/{dataset_name}/dataset_config.csv"
        dataset_config = pd.read_csv(dataset_config, sep=",", comment="'")
        # print(dataset_config.columns)
        fovs = dataset_config.loc[dataset_config[dataset_type]==1]
        datasetnum_per_fov = self.datasetnum // len(fovs)

        A = pd.read_csv(asd_file, sep="\s+", comment="'")
        dfs = None
        """
        for fov_id in fovs['FOV_ID']:

            df = A.loc[A['FOV_ID']==fov_id]
            if self.datasetnum > 0:
                df_sample = df.sample(n=datasetnum_per_fov, random_state=123456)
                if dfs is None: dfs = df_sample
                else: dfs = dfs.append(df_sample, ignore_index=False)
            else:
                if dfs is None: dfs = df
                else: dfs = dfs.append(df, ignore_index=False)
        """
        self.globbed_list_files_shuffle = []
        #for(baseX, baseY) in zip(dfs.base_x.values, dfs.base_y.values):
        for(baseX, baseY) in zip(A.base_x.values, A.base_y.values):
            datapath = f"{dataset_dir}/{dataset_name}/data/{baseX}_{baseY}.npy"
            self.globbed_list_files_shuffle.append(datapath)
        # print(len(self.globbed_list_files_shuffle))

        self.globbed_list_files_shuffle = np.array(self.globbed_list_files_shuffle)
        self.indices = np.array(range(self.globbed_list_files_shuffle.shape[0]))

    def load_dataset(
        self,
        batch_size,
        img_idx,
        repeat=1,
        is_train=False,
    ):
        def fixed_process(x):
            raw_name = x.numpy().decode()
            imgs = np.load(raw_name)
            imgs = np.expand_dims(imgs, axis=-1)

            # Assume square image
            h, w, _ = imgs.shape
            num_images = w // h
            assert ((w % h) == 0) or (num_images > 1)

            imgs_A = imgs[:, img_idx[0] * h : (img_idx[0] + 1) * h, :self.in_channels]
            # If merge_output_channels == True, sum through channels and merge them
            if self.merge_output_channels == True:
                imgs_B = imgs[:, img_idx[1] * h : (img_idx[1] + 1) * h, :]
                imgs_B = np.sum(imgs_B, axis=2, keepdims=True)
            else:
                imgs_B = imgs[:, img_idx[1] * h : (img_idx[1] + 1) * h, :self.out_channels]

            if self.tanh_norm == True:
                imgs_A = np.array(imgs_A).astype(np.float32) * 2 - 1
                imgs_B = np.array(imgs_B).astype(np.float32) * 2 - 1
            else:
                imgs_A = np.array(imgs_A).astype(np.float32)
                imgs_B = np.array(imgs_B).astype(np.float32)
                
            if self.is_test:
                return imgs_A, imgs_B, raw_name
            else: 
                return imgs_A, imgs_B

        # starttime = time.time()
        dataloader = tf.data.Dataset.from_tensor_slices(self.globbed_list_files_shuffle)

        #
        dataloader = dataloader.cache()
        if is_train: dataloader = dataloader.shuffle(self.datasetnum)
        #

        if self.is_test:
            dataloader = dataloader.map(
                lambda x: tf.py_function(
                    fixed_process, [x], [tf.float32, tf.float32, tf.string]
                )
            )
            dataloader = dataloader.map(
                lambda x, y, z: tf.py_function(
                    self.random_process,
                    [x, y, z],
                    [tf.float32, tf.float32, tf.string]
                )
            )
        else: 
            dataloader = dataloader.map(
                lambda x: tf.py_function(
                    fixed_process, [x], [tf.float32, tf.float32]
                )
            )
            dataloader = dataloader.map(
                lambda x, y: tf.py_function(
                    self.random_process,
                    [x, y],
                    [tf.float32, tf.float32]
                )
            )

        dataloader = dataloader.batch(
            batch_size, drop_remainder=True
        )
        
        return dataloader
    
    def load_batch(self, batch_size=1, img_idx=[0,1]):
        #
        def fixed_process(x):
            imgs = np.load(x)
            imgs = np.expand_dims(imgs, axis=-1)

            # Assume square image
            h, w, _ = imgs.shape
            num_images = w // h
            assert ((w % h) == 0) or (num_images > 1)

            imgs_A = imgs[:, img_idx[0] * h : (img_idx[0] + 1) * h, :self.in_channels]
            imgs_B = imgs[:, img_idx[1] * h : (img_idx[1] + 1) * h, :self.out_channels]

            if self.tanh_norm == True:
                imgs_A = np.array(imgs_A).astype(np.float32) * 2 - 1
                imgs_B = np.array(imgs_B).astype(np.float32) * 2 - 1
            else:
                imgs_A = np.array(imgs_A).astype(np.float32)
                imgs_B = np.array(imgs_B).astype(np.float32)
                
            return imgs_A, imgs_B
        
        #
        self.n_batches = int(self.datasetnum / batch_size)

        # print(self.globbed_list_files_shuffle)
        # print(self.indices)
        for i in range(self.n_batches):

            # batch = self.globbed_list_files_shuffle[i*batch_size : (i+1)*batch_size]
            batch_indices = self.indices[i*batch_size : (i+1)*batch_size]
            batch = self.globbed_list_files_shuffle[batch_indices]

            imgs_A, imgs_B, img_names = [],[],[]
            for img_file in batch:
                img_A, img_B = fixed_process(img_file)
                if self.is_test:
                    img_A, img_B, img_name = self.random_process(img_A, img_B, img_file)
                    img_names.append(img_name)
                else:
                    img_A, img_B = self.random_process(img_A, img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            # print(np.array(imgs_A).shape)
            # print(np.array(imgs_B).shape)
            
            if self.is_test:
                yield np.array(imgs_A), np.array(imgs_B), img_names
            else:
                yield np.array(imgs_A), np.array(imgs_B)
    
    def load_data_stream(self, img_idx=[0,1]):
        # Batch size: 1
        
        #
        def fixed_process(x):
            imgs = np.load(x)
            imgs = np.expand_dims(imgs, axis=-1)

            # Assume square image
            h, w, _ = imgs.shape
            num_images = w // h
            assert ((w % h) == 0) or (num_images > 1)

            imgs_A = imgs[:, img_idx[0] * h : (img_idx[0] + 1) * h, :self.in_channels]
            imgs_B = imgs[:, img_idx[1] * h : (img_idx[1] + 1) * h, :self.out_channels]

            if self.tanh_norm == True:
                imgs_A = np.array(imgs_A).astype(np.float32) * 2 - 1
                imgs_B = np.array(imgs_B).astype(np.float32) * 2 - 1
            else:
                imgs_A = np.array(imgs_A).astype(np.float32)
                imgs_B = np.array(imgs_B).astype(np.float32)
                
            return imgs_A, imgs_B
        

        for img_file in self.globbed_list_files_shuffle:
            img_A, img_B = fixed_process(img_file)
            if self.is_test:
                img_A, img_B, img_name = self.random_process(img_A, img_B, img_file)
            else:
                img_A, img_B = self.random_process(img_A, img_B)
            
            if self.is_test:
                yield img_A, img_B
            else:
                yield img_A, img_B
    
    def load_data_memory(self, img_idx=[0,1]):
        # Batch size: 1
        
        #
        def fixed_process(x):
            imgs = np.load(x)
            imgs = np.expand_dims(imgs, axis=-1)

            # Assume square image
            h, w, _ = imgs.shape
            num_images = w // h
            assert ((w % h) == 0) or (num_images > 1)

            imgs_A = imgs[:, img_idx[0] * h : (img_idx[0] + 1) * h, :self.in_channels]
            imgs_B = imgs[:, img_idx[1] * h : (img_idx[1] + 1) * h, :self.out_channels]

            if self.tanh_norm == True:
                imgs_A = np.array(imgs_A).astype(np.float32) * 2 - 1
                imgs_B = np.array(imgs_B).astype(np.float32) * 2 - 1
            else:
                imgs_A = np.array(imgs_A).astype(np.float32)
                imgs_B = np.array(imgs_B).astype(np.float32)
                
            return imgs_A, imgs_B
        

        imgs_A, imgs_B = [], []
        # imgs_name = []

        for img_file in self.globbed_list_files_shuffle:
            # imgs_name.append(img_file)
            img_A, img_B = fixed_process(img_file)
            if self.is_test:
                img_A, img_B, img_name = self.random_process(img_A, img_B, img_file)
            else:
                img_A, img_B = self.random_process(img_A, img_B)
            
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        
        return imgs_A, imgs_B
        # return imgs_A, imgs_B, imgs_name

