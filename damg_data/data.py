from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from damg_data.LOLdataset import *
from damg_data.eval_sets import *
from damg_data.SICE_blur_SID import *
from damg_data.fivek import *

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size, prior_config=None):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)


def get_lol_v2_training_set(data_dir,size, prior_config=None):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)


def get_training_set_blur(data_dir,size, prior_config=None):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)


def get_lol_v2_syn_training_set(data_dir,size, prior_config=None):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)


def get_SID_training_set(data_dir,size, prior_config=None):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)


def get_SICE_training_set(data_dir,size, prior_config=None):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)

def get_SICE_eval_set(data_dir, prior_config=None):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2(), prior_config=prior_config)

def get_eval_set(data_dir, prior_config=None):
    return DatasetFromFolderEval(data_dir, transform=transform2(), prior_config=prior_config)

def get_fivek_training_set(data_dir,size, prior_config=None):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size), prior_config=prior_config, crop_size=size)

def get_fivek_eval_set(data_dir, prior_config=None):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2(), prior_config=prior_config)
