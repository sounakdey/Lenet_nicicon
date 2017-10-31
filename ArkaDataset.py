import os
# import cv2
# from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

np.random.seed(666)

# text_feats = np.load('/home/sounak/Downloads/arka/F5K_Topics_textfeats.npy')
# img_feats = np.load('/home/sounak/Downloads/arka/F5K_Topics_googlenetfeat.npy')
# npz_labels = np.load('/home/sounak/Downloads/arka/F5K_Labels_Topics.npy')
# labels = npz_labels['labels']
# idx1 = list(range(len(labels)))
# idx1_tr = np.random.choice(idx1, int(0.8 * len(idx1)), replace=False).tolist()
# idx1_va = np.random.choice([x for x in idx1 if x not in idx1_tr], int(0.1 * len(idx1)), replace=False).tolist()
# idx1_te = [x for x in idx1 if x not in idx1_tr and x not in idx1_va]

class ArkaDataset(Dataset):

    def __init__(self, set_type='Train', transform=None):

        self.text_feats = np.load('/home/sounak/Downloads/arka/F5K_Topics_textfeats.npy')
        self.img_feats = np.load('/home/sounak/Downloads/arka/F5K_Topics_googlenetfeat.npy')
        npz_labels = np.load('/home/sounak/Downloads/arka/F5K_Labels_Topics.npy')
        self.labels = npz_labels['labels']
        idx1 = list(range(len(self.labels)))

        self.idx1_tr = np.random.choice(idx1, int(0.8 * len(idx1)), replace=False).tolist()
        self.idx1_va = np.random.choice([x for x in idx1 if x not in self.idx1_tr], int(0.1 * len(idx1)), replace=False).tolist()
        self.idx1_te = [x for x in idx1 if x not in self.idx1_tr and x not in self.idx1_va]
        self.set_type = set_type
        self.transform = transform

    def __getitem__(self, index):

        if self.set_type == 'Train':
            img_features = self.img_feats[self.idx1_tr][int(index)]
            text_features = self.text_feats[self.idx1_tr][int(index)]
            img_features = torch.from_numpy(img_features)
            text_features = torch.from_numpy(text_features)
                        # if self.transform is not None:
            #     img_features = self.transform(img_features)
            #     text_features = self.transform(text_features)
            labelss = self.labels[self.idx1_tr][int(index)]
        else:
            img_features = self.img_feats[self.idx1_te][int(index)]
            text_features = self.text_feats[self.idx1_te][int(index)]
            img_features = torch.from_numpy(img_features)
            text_features = torch.from_numpy(text_features)
                        # if self.transform is not None:
            #     img_features = self.transform(img_features)
            #     text_features = self.transform(text_features)
            labelss = self.labels[self.idx1_te][int(index)]

        return img_features, text_features, labelss

    def __len__(self):

        if self.set_type == 'Train':
            return len(self.idx1_tr)
        else:
            return len(self.idx1_te)
