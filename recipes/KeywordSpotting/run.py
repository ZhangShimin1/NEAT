'''
Description: 
version: 
Author: Shimin Zhang
Date: 2024-12-04 22:08:49
'''
import os
import sys
import yaml
import importlib
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from spikingjelly.activation_based.functional import reset_net

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from acouspike.models.network import BaseNet


def get_optim_scheduler(model, optim_conf, scheduler_conf):
    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam
    }
    optimizer_class = optimizers.get(optim_conf['type'])
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer type: {optim_conf['type']}")
    
    schedulers = {
        'step': lr_scheduler.StepLR,
        'exponential': lr_scheduler.ExponentialLR,
        'cosine': lr_scheduler.CosineAnnealingLR,
    }
    scheduler_class = schedulers.get(scheduler_conf['type'])
    if scheduler_class is None:
        raise ValueError(f"Unsupported scheduler type: {scheduler_conf['type']}")
    
    optimizer = optimizer_class(model.parameters(), lr=optim_conf['params']['learning_rate'])
    scheduler = scheduler_class(optimizer, **scheduler_conf['params'])

    return optimizer, scheduler

def prepare_data(data_conf):
    bs = data_conf["batch_size"]
    n_workers = data_conf["num_workers"]
    dataset = data_conf["dataset"]
    module = importlib.import_module(f'datasets.{dataset}')
    if dataset == "gsc":
        gsc_class = getattr(module, "Datasets")
        train_dataset = gsc_class(split="train", version=data_conf["version"], if_command=data_conf["if_command"], aug=data_conf["aug"])
        valid_dataset = gsc_class(split="valid", version=data_conf["version"], if_command=data_conf["if_command"], aug=False)
        test_dataset = gsc_class(split="test", version=data_conf["version"], if_command=data_conf["if_command"], aug=False)
        train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=n_workers, shuffle=True, pin_memory='cpu', sampler=None)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, num_workers=n_workers, shuffle=True, pin_memory='cpu', sampler=None)
        test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=n_workers, shuffle=True, pin_memory='cpu', sampler=None)
        in_dim = 40
        if data_conf["version"] == 1:
            if data_conf["if_command"]:
                out_dim = 10
            else:
                out_dim = 30
        else:
            if data_conf["if_command"]:
                out_dim = 14
            else:
                out_dim = 35
    elif dataset == "shd":
        if data_conf["if_spk"]:
            shd_class = getattr(module, "SpikingDatasets")
            train_dataset = shd_class(split="train")
            test_dataset = shd_class(split="test")
            in_dim = 700
        else:
            hd_class = getattr(module, "NonSpikingDatasets")
            train_dataset = hd_class(split="train", aug=data_conf["aug"])
            test_dataset = hd_class(split="test", aug=False)
            in_dim = 40
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers, collate_fn=train_dataset.generate_batch, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=n_workers, collate_fn=test_dataset.generate_batch, pin_memory=True)
        valid_loader = test_loader
        out_dim = 20
    elif dataset == "ssc":
        ssc_class = getattr(module, "SpikingDatasets")
        train_dataset = ssc_class(split="train")
        valid_dataset = ssc_class(split="valid")
        test_dataset = ssc_class(split="test")
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers, collate_fn=train_dataset.generate_batch, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=n_workers, collate_fn=test_dataset.generate_batch, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=n_workers, collate_fn=valid_dataset.generate_batch, pin_memory=True)
        in_dim = 700
        out_dim = 35

    return train_loader, valid_loader, test_loader, in_dim, out_dim


class Trainer:
    def __init__(self, config, loss_function):
        self.data_config = config["data"]
        self.train_loader, self.valid_loader, self.test_loader, in_dim, out_dim = prepare_data(self.data_config)
        
        self.model_config = config["model"]
        self.model = BaseNet(in_dim=in_dim, out_dim=out_dim, model_config=self.model_config)
        self.model = self.model.cuda()

        self.trainer_config = config["trainer"]
        self.epoch = self.trainer_config["num_train_epochs"]
        self.optim, self.scheduler = get_optim_scheduler(self.model, self.trainer_config["optimizer"], self.trainer_config["scheduler"])

        self.loss_function = loss_function

        self.save_dir = self.trainer_config["output_dir"]
        self.checkpoint_folder = self.save_dir + f'{self.data_config["dataset"]}/{self._get_time_now()}/checkpoint/'
        self.res_folder = self.save_dir + f'{self.data_config["dataset"]}/{self._get_time_now()}/res/'
        for folder in [self.checkpoint_folder, self.res_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _get_time_now(self):
        return datetime.now().strftime("%Y_%m_%d--%H_%M_%S")

    def _save_best_checkpoint(self, state):
        path = os.path.join(self.checkpoint_folder, 'best_valid_acc.pth.tar')
        torch.save(state, path)

    def _load_checkpoint(self, ckpt_choice="best"):
        if ckpt_choice == "best":
            path = os.path.join(self.checkpoint_folder, 'best_valid_acc.pth.tar')
            state_dict = torch.load(path)["state_dict"]
        else:
            raise NotImplementedError
        
        return state_dict
    
    def train(self):
        self.model.train()
        best_valid_acc = 0.
        for e in range(self.epoch):
            train_batch_loss, train_batch_acc = [], []
            for i, batch in enumerate(self.train_loader):
                loss, acc = self.training_step(batch, i)
                train_batch_loss.append(loss)
                train_batch_acc.append(acc)
            train_acc_avg = np.mean(train_batch_acc)
            train_loss_avg = np.mean(train_batch_loss)
            print(f"Train epoch: {e}, loss: {train_loss_avg}, accuracy: {train_acc_avg}")

            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self.validate()
                print(f"Validation epoch: {e}, loss: {val_loss}, accuracy: {val_acc}")
                if val_acc > best_valid_acc:
                    best_valid_acc = val_acc
                    saved_checkpoint = {
                        'epoch': e,
                        'acc': best_valid_acc,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optim.state_dict(),
                    }
                    self._save_best_checkpoint(saved_checkpoint)
            
    @torch.no_grad()
    def validate(self):
        valid_batch_loss, valid_batch_acc = [], []
        for i, batch in enumerate(self.valid_loader):
            loss, acc = self.evaluating_step(batch, i)
            valid_batch_loss.append(loss)
            valid_batch_acc.append(acc)
        valid_acc_avg = np.mean(valid_batch_acc)
        valid_loss_avg = np.mean(valid_batch_loss)

        return valid_loss_avg, valid_acc_avg

    @torch.no_grad()       
    def test(self):
        print("Begin testing...")
        test_batch_loss, test_batch_acc = [], []
        # load checkpoint with best valid acc for testing
        best_state_dict = self._load_checkpoint(ckpt_choice="best")
        self.model.load_state_dict(best_state_dict)
        self.model = self.model.cuda()
        for i, batch in enumerate(self.test_loader):
            loss, acc = self.evaluating_step(batch, i)
            test_batch_loss.append(loss)
            test_batch_acc.append(acc)
        test_acc = np.mean(test_batch_acc)
        test_loss = np.mean(test_batch_loss)
        print(f"Test ending: Loss: {test_loss}, Acc: {test_acc}")
        return test_loss, test_acc

    def training_step(self, batch, i):
        spect, target = batch
        x = spect.permute(1, 0, 2).cuda()
        y = target.cuda()
        # forward
        logits = self.model(x)
        loss = self.loss_function(logits, y)
        # backward
        self.optim.zero_grad()
        loss.backward()
        reset_net(self.model)
        self.optim.step()
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()
        loss, accuracy = loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

        return loss, accuracy
        
    def evaluating_step(self, batch, i):
        spect, target = batch
        x = spect.permute(1, 0, 2).cuda()
        y = target.cuda()
        # forward
        logits = self.model(x)
        loss = self.loss_function(logits, y)
        reset_net(self.model)
        # calculate acc
        _, preds = torch.max(logits, dim=1)
        accuracy = (preds == y).float().mean()
        loss, accuracy = loss.cpu().numpy(), accuracy.cpu().numpy()

        return loss, accuracy



if __name__ == "__main__":
    with open('conf/default.yaml', 'r') as file:
        configuration = yaml.safe_load(file)
    trainer = Trainer(config=configuration, loss_function=torch.nn.CrossEntropyLoss().cuda())
    trainer.train()
    trainer.test()