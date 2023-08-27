import logging
import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from .datasets import CIFAR10_truncated
from torch.utils.data import DataLoader, TensorDataset
import h5py
import gc

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append (tmp)
    return net_cls_counts

def record_part(y_test, train_cls_counts,test_dataidxs, logger):
    test_cls_counts = []

    for net_i, dataidx in enumerate(test_dataidxs):
        unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        test_cls_counts.append ( tmp)
        logger.debug('DATA Partition: Train %s; Test %s' % (str(train_cls_counts[net_i]), str(tmp) ))
    return


def _data_transforms_abcd():
    ABCD_MEAN = [0.5, 0.5, 0.5]
    ABCD_STD = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(121, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(ABCD_MEAN, ABCD_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ABCD_MEAN, ABCD_STD),
    ])

    return train_transform, valid_transform


def load_abcd_data_generator(X, y, site, batch_size):
    #train_transform, test_transform = _data_transforms_abcd()


    unique_sites = np.unique(site)

    # Define the train-test split ratio
    split_ratio = 0.2

    for site_value in unique_sites:
        site_indices = np.where(site == site_value)[0]
        
        site_test_size = int(len(site_indices) * split_ratio)
        site_train_size = len(site_indices) - site_test_size
        
        np.random.seed(42)
        np.random.shuffle(site_indices)
        
        site_X_train, site_X_test = X[site_indices[:site_train_size]], X[site_indices[site_train_size:]]
        site_y_train, site_y_test = y[site_indices[:site_train_size]], y[site_indices[site_train_size:]]
        site_train = np.full(site_train_size, site_value)
        site_test = np.full(site_test_size, site_value)
        
        # Convert to PyTorch tensors if needed
        X_train_tensor = torch.tensor(site_X_train)
        y_train_tensor = torch.tensor(site_y_train)
        site_train_tensor = torch.tensor(site_train)
        
        X_test_tensor = torch.tensor(site_X_test)
        y_test_tensor = torch.tensor(site_y_test)
        site_test_tensor = torch.tensor(site_test)
        
        yield (X_train_tensor, y_train_tensor, site_train_tensor), (X_test_tensor, y_test_tensor, site_test_tensor), torch.tensor(site_value)

        # Clear memory
        del X_train_tensor, y_train_tensor, site_train_tensor, X_test_tensor, y_test_tensor, site_test_tensor


def load_abcd_data(file_path):
   
    hdf5_file_name = file_path

    # Load data from the HDF5 file
    abcd_data = {}
    with h5py.File(hdf5_file_name, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            abcd_data[key] = hdf5_file[key][()]

    X = abcd_data['X']
    y = abcd_data['y']
    site = abcd_data['site']
    # Convert to PyTorch tensors if needed
    train_tensors = []
    test_tensors = []
    site_tensors = []

    batch_size = 128
    train_test_generator = load_abcd_data_generator(X, y, site, batch_size)
    for train_tuple, test_tuple, site_value in train_test_generator:
        # Process the current batch
        train_tensors.append(train_tuple)
        test_tensors.append(test_tuple)
        site_tensors.append(site_value)

        # Explicitly clear memory
        del train_tuple, test_tuple, site_value
        gc.collect()


    return train_tensors, test_tensors, site_tensors





def partition_data_abcd( datadir, partition, n_nets, alpha, logger):
    #X_train, X_test, y_train, y_test, site_train, site_test = load_abcd_data("")

    train_tensors, test_tensors, site_tensors = load_abcd_data(datadir)

    logger.info("*********partition data based on site***************")
    # Perform partitioning based on 'site' information

    # site_train = 21
    # unique_sites = np.unique(site_train)
    # n_sites = len(unique_sites)
    # n_nets = n_sites  # Number of clients will be the number of unique sites

    # net_dataidx_map = {i: np.where(site_train == site)[0] for i, site in enumerate(unique_sites)}

    # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    # return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    return train_tensors, test_tensors, site_tensors


def load_partition_data_abcd( data_dir, partition_method, partition_alpha, client_number, batch_size, logger):
    # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data_abcd(
    #                                                                                          data_dir,
    #                                                                                          partition_method,
    #                                                                                          client_number,
    #                                                                                          partition_alpha, logger)
    
    train_tensors, test_tensors, site_tensors = partition_data_abcd(                         data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha, logger)
   
    # Create dictionaries to store local dataset details
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    # Loop through each client
    for client_idx in range(21):
        # Get client-specific data tensors
        X_train_tensor, y_train_tensor, site_train_tensor = train_tensors[client_idx]
        X_test_tensor, y_test_tensor, site_test_tensor = test_tensors[client_idx]  

        # Convert tensors to Float32
        X_train_tensor = X_train_tensor.float()  # Convert to Float32
        y_train_tensor = y_train_tensor.float()  # Convert to Float32
        site_train_tensor = site_train_tensor.float()  # Convert to Float32

        X_test_tensor = X_test_tensor.float()  # Convert to Float32
        y_test_tensor = y_test_tensor.float()  # Convert to Float32
        site_test_tensor = site_test_tensor.float()  # Convert to Float32


        # Convert tensors to PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, site_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor, site_test_tensor)
        # Create custom dataloaders
        train_data_local = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_local = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Record local data number
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # Record partition details
    #record_part(y_test_tensor, traindata_cls_counts, test_dataidxs, logger)

    return None, None, None, None, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, 2 #traindata_cls_counts



def load_partition_data_abcd_rescale( data_dir, partition_method, partition_alpha, client_number, batch_size, logger):
    # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data_abcd(
    #                                                                                          data_dir,
    #                                                                                          partition_method,
    #                                                                                          client_number,
    #                                                                                          partition_alpha, logger)
    
    train_tensors, test_tensors, site_tensors = partition_data_abcd(                         data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha, logger)

    # # Create dictionaries to store local dataset details
    # data_local_num_dict = dict()
    # train_data_local_dict = dict()
    # test_data_local_dict = dict()
    # #N = 100  # Number of samples to select for each client

    # Merge all train_tensor X, y, and site independently
    all_X_train = torch.cat([train_tensors[i][0] for i in range(21)], dim=0)
    all_y_train = torch.cat([train_tensors[i][1] for i in range(21)], dim=0)
    all_site_train = torch.cat([train_tensors[i][2] for i in range(21)], dim=0)

    # Merge all test_tensor X, y, and site independently
    all_X_test = torch.cat([test_tensors[i][0] for i in range(21)], dim=0)
    all_y_test = torch.cat([test_tensors[i][1] for i in range(21)], dim=0)
    all_site_test = torch.cat([test_tensors[i][2] for i in range(21)], dim=0)

    # Convert tensors to Float32
    all_X_train = all_X_train.float()
    all_y_train = all_y_train.float()
    all_site_train = all_site_train.float()

    all_X_test = all_X_test.float()
    all_y_test = all_y_test.float()
    all_site_test = all_site_test.float()

    # Convert tensors to PyTorch datasets
    all_train_dataset = TensorDataset(all_X_train, all_y_train, all_site_train)
    all_test_dataset = TensorDataset(all_X_test, all_y_test, all_site_test)
    # Create dictionaries to store local dataset details
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    total_samples = len(all_X_train)  # Total number of samples

    # Calculate the number of samples per client
    samples_per_client = total_samples // client_number

    # Loop through each client
    for client_idx in range(client_number):
        # Calculate the start and end indices for selecting data
        start_idx = client_idx * samples_per_client
        end_idx = start_idx + samples_per_client
        
        # Select data indices for the current client from the merged train data
        selected_train_indices = list(range(start_idx, end_idx))
        
        # Select data indices for the current client from the merged test data
        test_start_idx = int(start_idx * 0.2)  # 20% test data
        test_end_idx = int(end_idx * 0.2)
        selected_test_indices = list(range(test_start_idx, test_end_idx))
        
         # Convert selected data indices to tensors
        X_train_tensor = all_X_train[selected_train_indices]
        y_train_tensor = all_y_train[selected_train_indices]
        site_train_tensor = all_site_train[selected_train_indices]
        
        X_test_tensor = all_X_test[selected_test_indices]
        y_test_tensor = all_y_test[selected_test_indices]
        site_test_tensor = all_site_test[selected_test_indices]
        
        # Convert tensors to Float32
        X_train_tensor = X_train_tensor.float()
        y_train_tensor = y_train_tensor.float()
        site_train_tensor = site_train_tensor.float()
        X_test_tensor = X_test_tensor.float()
        y_test_tensor = y_test_tensor.float()
        site_test_tensor = site_test_tensor.float()
        
        # Convert tensors to PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, site_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor, site_test_tensor)
        
        # Create custom dataloaders for both train and test data
        train_data_local = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_local = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Record local data number
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

        print("Done for client: " + str(client_idx))


    return None, None, None, None, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, 2 #traindata_cls_counts
