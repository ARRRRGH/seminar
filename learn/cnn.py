import pytorch_lightning as ptl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from collections import OrderedDict

try:
    from .cnn_lstm.convlstm import ConvLSTM
except ModuleNotFoundError:
    from seminar.cnn_lstm.convlstm import ConvLSTM


class LSTM(ptl.LightningModule):
    def __init__(self, classes, input_shape, train_image_folder, val_image_folder,
                 n_jobs, batch_size, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=True, bias=True, lr=1e-3):
        super().__init__()

        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder

        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.lr = lr

        self._classes = {val: key for key, val in enumerate(classes)}
        self.call_classes = np.vectorize(lambda entry: self._classes.get(entry, entry))
        self.classes = lambda tens: torch.Tensor(self.call_classes(tens)).to(torch.long)

        self.lstm = ConvLSTM(in_channels, hidden_channels, kernel_size, num_layers,
                             batch_first=batch_first, bias=bias, return_all_layers=False)

        padding = kernel_size[-1][0] // 2, kernel_size[-1][1] // 2
        self.conv = nn.Conv2d(in_channels=hidden_channels[-1], kernel_size=kernel_size[-1], out_channels=1,
                              padding=padding)

        self.linear_head = nn.Linear(in_features=int(np.prod(input_shape)), out_features=len(classes))

    def forward(self, x):
        h, c = self.lstm(x)
        output = self.conv(h.squeeze(2)[:, -1, :, :]).flatten(1)

        return self.linear_head(output)

    def training_step(self, batch, batch_idx):
        data, target = batch
        loss = F.cross_entropy(input=self.forward(data), target=self.classes(target))

        tqdm_dict = {'training_loss': loss, 'batch_idx': batch_idx}
        log = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        output = OrderedDict({'loss': loss})
        output.update(log)
        return output

    def train_dataloader(self):
        is_valid_file = lambda path: path.endswith('npy')
        loader = lambda path: torch.Tensor(np.load(path))
        dset = ImageFolder(self.train_image_folder, loader=loader, is_valid_file=is_valid_file)

        return DataLoader(dset, batch_size=self.batch_size, num_workers=self.n_jobs)

    def val_dataloader(self):
        is_valid_file = lambda path: path.endswith('npy')
        loader = lambda path: torch.Tensor(np.load(path))
        dset = ImageFolder(self.val_image_folder, loader=loader, is_valid_file=is_valid_file)

        return DataLoader(dset, batch_size=self.batch_size, num_workers=self.n_jobs)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
