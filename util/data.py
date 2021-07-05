import numpy as np
import os
import json
import logging
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot

class DF_Handler(object):
    def __init__(self, folder, filename):
        super(DF_Handler, self).__init__()

        if not os.path.exists(os.path.join('csv', folder)):
            os.makedirs(os.path.join('csv', folder))

        self.file_path = os.path.join('csv', folder, filename + '.csv')
        self.df = pd.DataFrame()
    
    def append(self, entry_dict):
        self.df = self.df.append(entry_dict, ignore_index=True)

    def save(self):
        self.df.to_csv(os.path.join(self.file_path), index=False)

class Log_Handler(object):
    def __init__(self, folder, filename):
        super(Log_Handler, self).__init__()

        self.folder = os.path.join('logs', folder)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.logger = logging.getLogger()
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s [%(levelname)s] - %(message)s',
            filename=os.path.join(self.folder, filename + '.log'))

    def append(self, line):
        print(line)
        self.logger.info(line)

def check_file(path):
    if os.path.isfile(path):
        return True
    else:
        raise RuntimeError('File not found: {}'.format(path))

def standardize_data(train_data, test_data):
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0)
    # handle constant features (they are not scaled)
    std[std == 0.] = 1.

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    return train_data, test_data

def load_dataset(config_path, standardize=False, maxmin=False,
                    normalize=False, flatten=True, dtype=torch.FloatTensor, split_size=0.9):
    check_file(config_path)

    with open(config_path) as json_file:
        config = json.load(json_file)

    check_file(config['train_data'])
    train_data, train_labels = torch.load(config['train_data'])

    if 'test_data' not in config:
        # create train/test split
        train_data, test_data, train_labels, test_labels = create_train_val_split(train_data, train_labels, train_size=split_size)
    else:
        check_file(config['test_data'])
        test_data, test_labels = torch.load(config['test_data'])

    if config['regression'] == False:
        train_labels[train_labels==-1] = 0
        test_labels[test_labels==-1] = 0

        train_data = train_data.type(dtype)
        test_data = test_data.type(dtype)
        if train_labels.shape[1] == 1:
            train_labels = train_labels.view(-1).type(torch.LongTensor)
            test_labels = test_labels.view(-1).type(torch.LongTensor)
            train_labels = one_hot(train_labels).type(dtype)
            test_labels = one_hot(test_labels).type(dtype)

    # Flatten the images
    if flatten:
        train_data = train_data.view(len(train_data), -1)
        test_data = test_data.view(len(test_data), -1)

    if maxmin:
        minx, _ = torch.min(train_data, 0)
        maxx, _ = torch.max(train_data, 0)
        ranges = maxx - minx
        ranges[ranges == 0] = 1 # to avoid NaN
        train_data = (train_data - minx) / ranges
        test_data = (test_data - minx) / ranges
        train_data = train_data * 2 - 1
        test_data = test_data * 2 - 1

    # Standardize (standard normalize every feature)
    if standardize:
        mean = train_data.mean(dim=0)
        std = train_data.std(dim=0)
        # handle constant features (they are not scaled)
        std[std == 0.] = 1.

        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

    # Normalize (x / ||x||)
    if normalize:
        train_data = train_data / train_data.norm(dim=1, keepdim=True)
        test_data = test_data / test_data.norm(dim=1, keepdim=True)

    return config['name'], train_data, test_data, train_labels, test_labels

def get_torch_dataset(data, labels=None):
    if labels is not None:
        return TensorDataset(data, labels)
    else:
        return TensorDataset(data)

def get_dataloader(data, labels=None, batchsize=3000, shuffle=True, drop_last=False):
    return DataLoader(
        get_torch_dataset(data, labels),
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        # drop last batch if incomplete
        drop_last=drop_last
    )

def create_train_val_split(train_data, train_labels, train_size=0.8):
    """
    Splits the training data into training and validation data.
    train_size is the ratio of the training set.
    """
    train_size = int(train_size * len(train_labels))
    perm = torch.randperm(len(train_data), )
    train_idxs = perm[:train_size]
    val_idxs = perm[train_size:]
    
    X_train = train_data[train_idxs]
    X_val = train_data[val_idxs]
    y_train = train_labels[train_idxs]
    y_val = train_labels[val_idxs]

    return X_train, X_val, y_train, y_val

def pad_data_pow_2(data, offset=0):
    """
    Pads the input with zeros s.t. d=2**i - offset.
    """

    d_new = int(2**np.ceil(np.log2(data.shape[1]+offset)))
    placeholder = torch.zeros(len(data), d_new-offset, device=data.device)
    placeholder[:, :data.shape[1]] = data
    
    return placeholder

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5data, features, labels):
        self.h5data = h5data
        self.features = h5data[features]
        self.labels = h5data[labels]
        
    def __getitem__(self, index):
        # self.features.get_data("data", index)
        return (torch.from_numpy(self.features[index, ...]), torch.from_numpy(self.labels[index, ...]))

    def __len__(self):
        return len(self.labels)