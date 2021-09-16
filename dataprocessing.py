from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from collections import Counter
import scipy.io as scio
import gc

import torch
import sys
import time
import joblib

import matplotlib.pyplot as plt

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict
class data_loader(Dataset):
    def __init__(self, samples, labels,args=None):

        self.samples = samples
        self.labels = labels
        self.args = args


    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.args.dataset == 'cifar10' :
            apply_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            sample = Image.fromarray(sample)
            sample = apply_transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


def get_Users_Data(args,train_dataset,noise_lst=[]):
    eachUser_dat = []


    user_groups = cifar_assign(train_dataset, args.num_users)
    noise_chart = [[] for i in range(args.num_users)]
    for i in range(args.num_users):

        idxs = user_groups[i]
        idxs = idxs.astype(int)
        X_u, Y_u = train_dataset.samples[idxs], train_dataset.labels[idxs]
        # noise_chart[i] = noise_lst[idxs]
        eachUser_dat.append(data_loader(X_u, Y_u,args))

    return eachUser_dat,noise_chart


def cifar_assign(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = np.array(list(dict_users[i]))
    return dict_users


def bench_assign(dataset):

    len_dataset = len(dataset)
    num_items = int(len_dataset / 20)
    all_idxs =[i for i in range(len_dataset)]
    np.random.seed(42)

    sub_set = np.random.choice(all_idxs, num_items,
                                         replace=False)
    return sub_set

def get_dataset(args):

    train_data = []
    train_label = []
    if args.dataset == 'cifar10':

        root_dir = 'data/cifar-10/cifar-10-batches-py'
        for n in range(1, 6):
            dpath = '%s/data_batch_%d' % (root_dir, n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            train_label = train_label + data_dic['labels']
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))
        train_dataset = data_loader(train_data, np.array(train_label),args)

        test_dic = unpickle('%s/test_batch' % root_dir)
        test_data = test_dic['data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic['labels']
        test_dataset = data_loader(test_data, np.array(test_label),args)

    return train_dataset, test_dataset

