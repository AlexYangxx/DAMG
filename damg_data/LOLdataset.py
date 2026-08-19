
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from damg_data.util import *
from damg_data.prior_utils import prepare_training_sample
from torchvision import transforms as t

    
class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, prior_config=None, crop_size=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.prior_config = prior_config
        self.crop_size = crop_size
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.low_dir = self.data_dir + '/low'
        self.high_dir = self.data_dir + '/high'
        self.data_filenames = [join(self.low_dir, x) for x in listdir(self.low_dir) if is_image_file(x)]
        self.data_filenames2 = [join(self.high_dir, x) for x in listdir(self.high_dir) if is_image_file(x)]
        self.data_filenames.sort()
        self.data_filenames2.sort()

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
        if self.prior_config and self.prior_config.enabled:
            im1, im2, priors = prepare_training_sample(
                im1,
                im2,
                self.data_filenames[index],
                crop_size=self.crop_size,
                prior_config=self.prior_config,
                reference_root=self.low_dir,
            )
            return im1, im2, file1, file2, priors
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)

    
class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, prior_config=None, crop_size=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.prior_config = prior_config
        self.crop_size = crop_size
        self.low_dir = self.data_dir + '/Low'
        self.high_dir = self.data_dir + '/Normal'
        self.data_filenames = [join(self.low_dir, x) for x in listdir(self.low_dir) if is_image_file(x)]
        self.data_filenames2 = [join(self.high_dir, x) for x in listdir(self.high_dir) if is_image_file(x)]
        self.data_filenames.sort()
        self.data_filenames2.sort()

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
        if self.prior_config and self.prior_config.enabled:
            im1, im2, priors = prepare_training_sample(
                im1,
                im2,
                self.data_filenames[index],
                crop_size=self.crop_size,
                prior_config=self.prior_config,
                reference_root=self.low_dir,
            )
            return im1, im2, file1, file2, priors
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)      
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7 
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)



class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None, prior_config=None, crop_size=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.prior_config = prior_config
        self.crop_size = crop_size
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.low_dir = self.data_dir + '/Low'
        self.high_dir = self.data_dir + '/Normal'
        self.data_filenames = [join(self.low_dir, x) for x in listdir(self.low_dir) if is_image_file(x)]
        self.data_filenames2 = [join(self.high_dir, x) for x in listdir(self.high_dir) if is_image_file(x)]
        self.data_filenames.sort()
        self.data_filenames2.sort()

    def __getitem__(self, index):
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
        if self.prior_config and self.prior_config.enabled:
            im1, im2, priors = prepare_training_sample(
                im1,
                im2,
                self.data_filenames[index],
                crop_size=self.crop_size,
                prior_config=self.prior_config,
                reference_root=self.low_dir,
            )
            return im1, im2, file1, file2, priors
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        return len(self.data_filenames)



    
