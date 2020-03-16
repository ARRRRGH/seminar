import pytorch_lightning as ptl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

try:
    from .cnn_lstm.convlstm import ConvLSTM
except ModuleNotFoundError:
    from seminar.cnn_lstm.convlstm import ConvLSTM


class TrainLSTM(ptl.LightningModule):
    def __init__(self, classes, input_shape, train_image_folder, n_jobs, batch_size, in_channels, hidden_channels,
                 kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False, lr=1e-3):
        super().__init__()

        self.train_image_folder = train_image_folder
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.lr = lr
        self.classes = {val: key for key, val in enumerate(classes)}

        self.lstm = ConvLSTM(in_channels, hidden_channels, kernel_size, num_layers,
                             batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=hidden_channels[-1], kernel_size=kernel_size[-1], out_channels=1,
                              padding=padding)

        self.linear_head = nn.Linear(in_features=int(np.prod(input_shape[1:])), out_features=nr_classes)

    def forward(self, x):
        output = self.lstm(x)
        output = self.conv(output).flatten()

        return self.linear_head(output)

    def training_step(self, batch, batch_idx):
        data, target = batch
        return F.cross_entropy(input=self.forward(batch), target=self.classes[target])

    def train_dataloader(self):
        loader = lambda path: torch.Tensor(np.load(path))
        ImageFolder(self.train_image_folder, loader=loader)

        return DataLoader(ImageFolder, batch_size=self.batch_size, num_workers=self.n_jobs)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
