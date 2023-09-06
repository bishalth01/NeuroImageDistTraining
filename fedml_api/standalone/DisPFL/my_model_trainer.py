import copy
import logging
import time

import numpy as np
import pdb
import torch
from torch import nn

from fedml_api.model.cv.cnn_meta import Meta_net
import h5py
import torch.nn.functional as F


try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger = None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def init_masks(self, params, sparsities):
        masks ={}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] =1
        return masks

    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse = 0.5):
        spasities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.args.dense_ratio
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            self.logger.info('initialize by ERK')
            total_params = 0
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()

            density = sparse
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(params[name].shape) / np.prod(params[name].shape)
                                                  ) ** self.args.erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if name in dense_layers:
                    spasities[name] = 0
                else:
                    spasities[name] = (1 - epsilon * raw_probabilities[name])
        return spasities

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    # def screen_gradients(self, train_data, device):
    #     model = self.model
    #     model.to(device)
    #     model.eval()
    #     # # # train and update
    #     criterion = nn.CrossEntropyLoss().to(device)
    #     # # sample one epoch  of data
    #     model.zero_grad()
    #     (x, labels) = next(iter(train_data))
    #     x, labels = x.to(device), labels.to(device)
    #     log_probs = model.forward(x)
    #     loss = criterion(log_probs, labels.long())
    #     loss.backward()
    #     gradient={}
    #     for name, param in model.named_parameters():
    #         gradient[name] = param.grad.to("cpu")
    #     return gradient

    def screen_gradients(self, train_data, device):
        model = self.model
        model.to(device)
        model.eval()
        # # # train and update
        criterion = nn.BCEWithLogitsLoss().to(device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels, site) = next(iter(train_data))
        
        #For 3DConv Network
        #x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor
        x = x.to(device)  # Convert to tensor
        x = x.unsqueeze(1)

        x, labels = x.to(device), labels.to(device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient={}
        for name, param in model.named_parameters():
            gradient[name] = param.grad.to("cpu")
        return gradient


    # def train(self, train_data,  device,  args, round):
    #     # torch.manual_seed(0)
    #     model = self.model
    #     model.to(device)
    #     model.train()
    #     # train and update
    #     criterion = nn.CrossEntropyLoss().to(device)
    #     if args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
    #     for epoch in range(args.epochs):
    #         epoch_loss = []
    #         for batch_idx, (x, labels) in enumerate(train_data):
    #             x, labels = x.to(device), labels.to(device)
    #             model.zero_grad()
    #             log_probs = model.forward(x)
    #             loss = criterion(log_probs, labels.long())
    #             loss.backward()
    #             # to avoid nan loss
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
    #             optimizer.step()
    #             epoch_loss.append(loss.item())
    #             for name, param in self.model.named_parameters():
    #                 if name in self.masks:
    #                     param.data *= self.masks[name].to(device)
    #         self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #             self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

    def load_data_chunks_batch(self, indexes):
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
        return X_batch, y_batch

    def train(self, train_data,  device,  args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.BCEWithLogitsLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
        for epoch in range(args.epochs):
            epoch_loss = []
            #for batch_idx, (x, labels) in enumerate(train_data):
            for x, labels, _ in train_data:
                #For 3DConv Network
                x, labels = self.load_data_chunks_batch(x)
                x = x.unsqueeze(1)

                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.unsqueeze(1).float())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())

                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        param.data *= self.masks[name].to(device)
                

            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))




    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0
        }
        criterion = nn.BCEWithLogitsLoss().to(device)

        with torch.no_grad():
            for x, target, _ in test_data:
                #For 3DConv Network
                #x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor
                #x = x.to(device)  # Convert to tensor
                x, target = self.load_data_chunks_batch(x)
                x = x.unsqueeze(1)

                x = x.to(device)
                target = target.to(device)
                pred = F.sigmoid(model(x))
                loss = criterion(pred, target.unsqueeze(1).float())

                final_preds = (pred >= 0.5).float().squeeze(1)
                correct = (final_preds == target).float().sum()
                accuracy = correct / len(target)

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics
    # def test(self, test_data, device, args):
    #     model = self.model

    #     model.to(device)
    #     model.eval()

    #     metrics = {
    #         'test_correct': 0,
    #         'test_acc':0.0,
    #         'test_loss': 0,
    #         'test_total': 0
    #     }

    #     criterion = nn.CrossEntropyLoss().to(device)

    #     with torch.no_grad():
    #         for batch_idx, (x, target) in enumerate(test_data):
    #             x = x.to(device)
    #             target = target.to(device)
    #             pred = model(x)
    #             loss = criterion(pred, target.long())

    #             _, predicted = torch.max(pred, -1)
    #             correct = predicted.eq(target).sum()

    #             metrics['test_correct'] += correct.item()
    #             metrics['test_loss'] += loss.item() * target.size(0)
    #             metrics['test_total'] += target.size(0)
    #             metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
    #     return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

