#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:35:43 2020

@author: kdh
"""
import pandas as pd
import os
from PIL import Image

import numpy as np
from scipy.stats import entropy

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
import torchvision.datasets as dset
import torchvision.transforms as transforms


def inception_score(imgs, model, cuda=True, batch_size=32, resize=False, splits=1):
    """
    Computes the inception score of the generated images imgs

    Arguments
    ---------
    imgs : torch.tensor
        Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda: bool
        whether or not to run on GPU
    batch_size : int
        batch size for feeding into Inception v3
    splits : int
        number of splits
    """

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = model(x)  # [batch_size, embeding_dim]
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch['image']
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
    # Original CIFAR10 : (9.672780722701322, 0.14991599396927277)


class CustomDataset(torch.utils.data.Dataset):
    """
    Pytorch dataloader for custom dataset
    
    Arguments
    ---------
    csv_file : string
        Path to the csv file with annotations.
     root_dir : string
        Directory with all the images.
    transform : callable, optional
        Optional transform to be applied on a sample.    
    """

    def __init__(self, csv_file, root_dir, transform=None, inFolder=None, landmarks=False):
        self.training_sheet = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if inFolder.any() == None:
            self.inFolder = np.full((len(self.training_sheet),), True)
        
        self.loc_list = np.where(inFolder)[0]
        self.infold = inFolder
        
    def __len__(self):
        return  np.sum(self.infold*1)     

    def __getitem__(self, idx):
        idx = self.loc_list[idx] 
#        label = self.training_sheet.iloc[idx,1]

        img_name = os.path.join(self.root_dir,
                                self.training_sheet.iloc[idx,0])
        
        image = Image.open(img_name)
        sample = image
        
        if self.transform:
            sample = self.transform(sample)
        #return {'image': sample, 'label': label}
        return {'image': sample}


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)
    

if __name__ == '__main__':

    cuda=True
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    all_path = '/path/all.csv'
    training_sheet = pd.read_csv(all_path)
    training_sheet_split = pd.DataFrame(training_sheet.subDirectory_filePath.str.split("/").tolist(),columns = ['subfolder', 'label'])
    folders = list(map(int,training_sheet_split.subfolder))
    folder_list = list(range(0,500))
    inFolder_train = np.isin(folders, folder_list)
    
    cifar = CustomDataset(csv_file=all_path,
                       root_dir='/path/subfolder/',
                       transform=transforms.Compose([
#                                 transforms.Scale(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]), inFolder=inFolder_train
    )
    
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

#    cifar = dset.CIFAR10(root='/path', download=True,
#                             transform=transforms.Compose([
#                                 transforms.Scale(32),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                             ])
#    )

    print("Calculating Inception Score...")
    print(inception_score(cifar, inception_model, batch_size=1, resize=True, splits=10))
