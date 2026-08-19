
import os
import torch.utils.data as data
from os import listdir
from os.path import join
from damg_data.util import *
from damg_data.prior_utils import prepare_eval_sample
import torch.nn.functional as F

class SICEDatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None, prior_config=None):
        super(SICEDatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform
        self.data_dir = data_dir
        self.prior_config = prior_config

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.prior_config and self.prior_config.enabled:
            input, priors, h, w = prepare_eval_sample(
                input,
                self.data_filenames[index],
                prior_config=self.prior_config,
                reference_root=self.data_dir,
                pad_to_factor=8,
            )
            return input, file, h, w, priors

        if self.transform:
            input = self.transform(input)
            factor = 8
            h, w = input.shape[1], input.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
        return input, file, h, w

    def __len__(self):
        return len(self.data_filenames)
    
    
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None, prior_config=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform
        self.data_dir = data_dir
        self.prior_config = prior_config

    def __getitem__(self, index):
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.prior_config and self.prior_config.enabled:
            input, priors, _, _ = prepare_eval_sample(
                input,
                self.data_filenames[index],
                prior_config=self.prior_config,
                reference_root=self.data_dir,
                pad_to_factor=None,
            )
            return input, file, priors

        if self.transform:
            input = self.transform(input)
        return input, file

    def __len__(self):
        return len(self.data_filenames)
