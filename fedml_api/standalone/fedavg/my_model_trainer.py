import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn
from fedml_api.model.cv.cnn_meta import Meta_net
import h5py
import torch.nn.functional as F

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def set_masks(self, masks):
        self.masks=masks
        # self.model.set_masks(masks)

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict
    
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
    #             # self.logger.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
    #             #            100. * (batch_idx + 1) / len(train_data), loss.item()))
    #             epoch_loss.append(loss.item())

    #         self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #             self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

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
                

            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))




    # def test(self, test_data, device, args):
    #     model = self.model

    #     model.to(device)
    #     model.eval()

    #     metrics = {
    #         'test_correct': 0,
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
    #     return metrics

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



    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

