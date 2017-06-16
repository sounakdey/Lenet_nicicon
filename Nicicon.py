import os
import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np


def default_image_loader(path):
    imag = Image.fromarray(cv2.resize(np.array(Image.open(path).convert('L')), (100, 100)))
    return imag


class NicIcon(Dataset):
    """Data-set wrapping images and target labels for NicIcon.

    Arguments:
      base_path = Directory where both the train and test data-set is
      img_ext = extension of the image
      transforms = to tensor and others if needed
    """

    def __init__(self, base_path, train=False, img_ext='.pbm', transform=None, loader=default_image_loader):
        if train:
            self.base_path = os.path.join(base_path, 'train')
        else:
            self.base_path = os.path.join(base_path, 'test')

        self.img_ext = img_ext
        self.train = train
        self.img_ext = img_ext
        self.transform = transform
        self.loader = loader

        self.fnames = [fp for (dirpath, dirnames, fname) in os.walk(self.base_path) for fp in fname if
                       fp.endswith('.pbm')]

    def __getitem__(self, index):
        img = self.loader(os.path.join(self.base_path, self.fnames[int(index)]))
        label = int(self.fnames[int(index)].split('_')[0]) - 1

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.fnames)
