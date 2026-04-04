import os
import time
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import matplotlib.pyplot as plt

from cgcnn_model.cgcnn.data_process import CIFData
from cgcnn_model.cgcnn.data_process import collate_pool, get_data_loader
from cgcnn_model.cgcnn.model import CrystalGraphConvNet


class Crystal_Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_data()
        self.build_model()

        np.set_printoptions(precision=4)
        torch.set_printoptions(precision=4)

    def load_data(self):
        self.dataset = CIFData(label_dir=self.cfg.label_dir, properties=self.cfg.properties,
                               struct_dir=self.cfg.structure_dir, aug=self.cfg.augmentation)
        collate_fn = collate_pool
        self.data_loader = get_data_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.workers,
            pin_memory=self.cfg.cuda)

        # obtain target value normalizer

        sample_data_list = [self.dataset[i] for i in sample(range(len(self.dataset)), min(200, len(self.dataset)))]
        _, sample_target, _ = collate_pool(sample_data_list)
        self.normalizer = Normalizer(sample_target)

    def build_model(self):
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        self.model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                         atom_fea_len=self.cfg.atom_fea_len,
                                         n_conv=self.cfg.n_conv,
                                         h_fea_len=self.cfg.h_fea_len,
                                         n_h=self.cfg.n_h, outsize=self.cfg.out_size, dropout=self.cfg.dropout)

        if self.cfg.cuda:
            self.model.cuda()

        # define loss func and optimizer
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), self.cfg.lr,
                                    weight_decay=self.cfg.weight_decay)

        self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg.lr_milestones, gamma=0.1)

    def train(self):

        for epoch in range(self.cfg.epochs):
            # train for one epoch
            self.train_epoch(epoch)

        # test best model
        print('---------Evaluate Model on Test Set---------------')

        root_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(root_dir, "save_dir", self.cfg.save_dir, )
        save_path = os.path.join(save_dir, "model.pth")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'normalizer': self.normalizer.state_dict(),
            'args': vars(self.cfg)
        }, filename=save_path)

    def train_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        mae_errors = AverageMeter()

        # switch to train mode
        self.model.train()
        end = time.time()

        for i, (input, target, _) in enumerate(self.data_loader):
            # measure data loading time
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            data_time.update(time.time() - end)

            if self.cfg.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
            print(input[0].device, target.device, self.model.atom_embedding.weight.device)


            # normalize target

            target_normed = self.normalizer.norm(target)

            target_var = Variable(target_normed.cuda(non_blocking=True))

            # compute output
            output = self.model(*input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss

            mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

            # compute gradient and do SGD step

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch % self.cfg.print_freq == self.cfg.print_freq - 1:
            print('Epoch: [{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'MAE {mae_errors.avg}'.format(
                epoch + 1, batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors))

    def predict(self, eval_times=50):

        if os.path.isfile(self.cfg.model_path):
            print("=> loading checkpoint '{}'".format(self.cfg.model_path))
            checkpoint = torch.load(self.cfg.model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            print("=> no checkpoint found at '{}'".format(self.cfg.model_path))
            return 0

        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        predictions = pd.DataFrame(data=None)

        L = len(self.data_loader)
        print("Start prediction:\n")

        with torch.no_grad():
            for i, (input, target, batch_cif_ids) in enumerate(self.data_loader):

                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])

                outputs = np.zeros((eval_times, len(target), len(self.cfg.properties)))
                for j in range(eval_times):
                    output = self.model(*input_var)
                    output = self.normalizer.denorm(output.data.cpu())
                    outputs[j, :] = output

                values_mean = np.mean(outputs, axis=0)
                values_std = np.std(outputs, axis=0)

                pred = {"Crystals": batch_cif_ids}

                for n, p in enumerate(self.cfg.properties):
                    pred["v_{}_mean".format(p)] = values_mean[:, n]
                    pred["v_{}_std".format(p)] = values_std[:, n]

                pred = pd.DataFrame(data=pred)

                predictions = predictions.append(pred)

                print("Progress {}/{} {:.2f}%".format((i + 1) * self.cfg.batch_size, L * self.cfg.batch_size,
                                                      (i + 1) * 100 / L))

        print("=> prediction completed")

        save_dir = os.path.dirname(self.cfg.model_path)

        save_path = os.path.join(save_dir, "predictions.csv")
        predictions.to_csv(save_path, header=True, index=False)

        return predictions

    def vis_recomm(self, preds):

        # -----------------visualization----------------#

        mean = preds["Values_mean"]
        std = preds["Values_std"]
        ub = mean + std
        lb = mean - std

        plt.plot(ub, c="cyan")
        plt.plot(lb, c="cyan", label="range")
        plt.plot(mean, c="black", label="mean", linewidth=1)

        plt.xlabel("Catalyst Index")
        plt.ylabel("Predicted Value")
        plt.title("Catalyst Property")
        plt.legend()
        plt.show()

        # -----------------recommendation----------------#

        large_var = preds.nlargest(10, 'Values_std')['Crystals']
        large_mean = preds.nlargest(10, 'Values_mean')['Crystals']

        print("""\nRecommendations:
              ---High Uncertainty---
              {}\n
              ---High Predicted Value---
              {}\n
              """.format(large_var.to_string(),
                         large_mean.to_string()))

        return 0


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """

    dif = torch.abs(target - prediction)

    return torch.mean(dif, dim=0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
