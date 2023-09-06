import copy
import logging
import math

import numpy as np
import pdb
import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Subset
import h5py
from fedml_api.standalone.sailentgrads.snip import get_snip_scores, get_mean_sailency_scores

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    #This case handles for IterSNIP as well as stratified sampling. If IterSNIP Iteration is 1, that is the default SNIP that was used
    #in the paper.
    def generate_sailency_scores_from_each_client(self, itersnip_iteration=20, stratified_sampling=False):
        local_training_data = self.local_training_data
        final_sailency_list=[]
        if stratified_sampling:
            y = np.concatenate(np.array([label for _, label in local_training_data]))
            X = np.concatenate([image for image, _ in local_training_data])
            #splitter = StratifiedShuffleSplit(n_splits=itersnip_iteration, test_size=0.1, random_state=42)
            splitter = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)
            for train_index, _ in splitter.split(X, y):
                train_mini_batch = [local_training_data.dataset[i] for i in train_index]
                train_X = torch.cat([torch.Tensor(x).unsqueeze(0) for x, _ in train_mini_batch], dim=0)
                train_y = torch.Tensor([y for _, y in train_mini_batch]).long()
                local_abs_grads = get_snip_scores(self.model_trainer,(train_X, train_y))
                final_sailency_list.append(local_abs_grads)
        else:
            for i in range(itersnip_iteration):
                #local_training_data = self.load_data_chunks(local_training_data)
                train_mini_batch = next(iter(local_training_data))
                train_mini_batch = self.load_data_chunks(train_mini_batch)
                local_abs_grads = get_snip_scores(self.model_trainer,train_mini_batch)
                final_sailency_list.append(local_abs_grads)
            
        mean_sailency_score = get_mean_sailency_scores(final_sailency_list)
        return mean_sailency_score
    
    def load_data_chunks(self, train_mini_batch):
        indexes = train_mini_batch[0]
        hdf5_file_name = "/data/users2/bthapaliya/NeuroimageDistributedFL/SailentWeightsDistributedFL/alldatain8bitsnormalized.h5"
        # Open the HDF5 file
        with h5py.File(hdf5_file_name, 'r') as hdf5_file:
            num_samples = len(hdf5_file['X'])  # Assuming 'X' is your dataset name
            mask_indexes = indexes.numpy().astype(int)
            mask_indexes = np.sort(mask_indexes)

            # Load the data batch
            X_batch = hdf5_file['X'][mask_indexes]
            y_batch = hdf5_file['y'][mask_indexes]
        
        X_batch = torch.tensor(X_batch,dtype=torch.float32, device="cuda")
        y_batch = torch.tensor(y_batch,dtype=torch.float32, device="cuda")
        train_mini_batch[0] = X_batch
        train_mini_batch[1] = y_batch
        return train_mini_batch

    
    

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global,round, masks):

        num_comm_params = self.model_trainer.count_communication_params(w_global)
        self.model_trainer.set_model_params(w_global)

        model_sparsity = self.model_trainer.get_model_sps()

        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train(self.local_training_data, self.device, self.args, round, masks)
        weights = self.model_trainer.get_model_params()
        

        #-----------------------Calculating FLOPS and Communication Parameters --------------------------------
        training_flops = self.args.epochs * self.local_sample_number #* self.model_trainer.count_training_flops_per_sample()
        #sparse_flops_per_data = self.model_trainer.count_training_flops_per_sample()
        #full_flops = self.model_trainer.count_full_flops_per_sample()
        #self.logger.info("training flops per data {}".format(sparse_flops_per_data))
        #self.logger.info("full flops for search {}".format(full_flops))
        # we train the data for `self.args.epochs` epochs, and forward one epoch of data with full density to screen gradient.
        #training_flops = self.args.epochs*self.local_sample_number*sparse_flops_per_data+ self.args.batch_size* full_flops
        num_comm_params += self.model_trainer.count_communication_params(weights)
        self.logger.info("communication parameters for search {}".format(num_comm_params))
        #---------------------------------END Flops and Communication Params-----------------------------------

        return  weights,training_flops,num_comm_params


    #Checking for the basic metrics
    def local_test(self, w_per, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        
        self.model_trainer.set_model_params(w_per)
        
        model_sparsity = self.model_trainer.get_model_sps()
        #print("Sparsity before testing in local is : " + str(model_sparsity))
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
