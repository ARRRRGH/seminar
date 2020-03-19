import pytorch_lightning as ptl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from collections import OrderedDict
from torchvision import models


try:
    from .cnn_lstm.convlstm import ConvLSTM
except ModuleNotFoundError:
    from seminar.cnn_lstm.convlstm import ConvLSTM


class LSTM(ptl.LightningModule):
    def __init__(self, classes, input_shape, train_image_folder, val_image_folder,
                 n_jobs, batch_size, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=True, bias=True, lr=1e-3, epoch_size=None):
        super().__init__()

        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder

        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.n_jobs = n_jobs
        self.lr = lr

        self._classes = {val: key for key, val in enumerate(classes)}
        self.call_classes = np.vectorize(lambda entry: self._classes.get(entry, entry))
        self.classes = lambda arr: torch.Tensor(self.call_classes(arr)).to(torch.long)

        self.lstm = ConvLSTM(in_channels, hidden_channels, kernel_size, num_layers,
                             batch_first=batch_first, bias=bias, return_all_layers=False)

        padding = kernel_size[-1][0] // 2, kernel_size[-1][1] // 2
        self.conv = nn.Conv2d(in_channels=hidden_channels[-1], kernel_size=kernel_size[-1], out_channels=1,
                              padding=padding)

        self.linear_head = nn.Linear(in_features=int(np.prod(input_shape)), out_features=len(classes))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h, c = self.lstm(x)
        h, c = h[0], c[0]

        output = []
        for t in range(h.shape[1]):
            output.append(self.conv(h[:, t, :, :, :]).flatten(1))

        output = torch.stack(output, dim=1)
        return self.linear_head(torch.max(output, dim=1)[0])

    def training_step(self, batch, batch_idx):
        data, target = batch
        loss = self.loss(input=self.forward(data), target=self.classes(target))

        tqdm_dict = {'training_loss': loss, 'batch_idx': batch_idx}
        log = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        output = OrderedDict({'loss': loss})
        output.update(log)
        return output

    def validation_step(self, batch, batch_idx):
        data, target = batch
        loss = self.loss(input=self.forward(data), target=self.classes(target))

        tqdm_dict = {'val_loss': loss, 'batch_idx': batch_idx}
        log = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        output = OrderedDict({'val_loss': loss})
        output.update(log)
        return output

    def _get_dataloader(self, path):
        is_valid_file = lambda path: path.endswith('npy')
        loader = lambda path: torch.Tensor(np.load(path))

        dset = ImageFolder(path, loader=loader, is_valid_file=is_valid_file)

        if self.epoch_size is not None:
            dset = Subset(dset, np.random.choice(len(dset), self.epoch_size, replace=False))

        return DataLoader(dset, batch_size=self.batch_size, num_workers=self.n_jobs)

    def train_dataloader(self):
        return self._get_dataloader(self.train_image_folder)

    def val_dataloader(self):
        return self._get_dataloader(self.val_image_folder)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024):
        super(EncoderCNN, self).__init__()

        # get the pretrained densenet model
        self.densenet = models.densenet121(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)

        # add another fully connected layer
        self.embed = nn.Linear(in_features=1024, out_features=embed_size)

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # activation layers
        self.prelu = nn.PReLU()

    def forward(self, images):

        # get the embeddings from the densenet
        densenet_outputs = self.dropout(self.prelu(self.densenet(images)))

        # pass through the fully connected
        embeddings = self.embed(densenet_outputs)

        return embeddings


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):

        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()

        # embed the captions
        captions_embed = self.embed(captions)

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.fc_out(hidden_state)

            # build the output tensor
            outputs[:, t, :] = out

        return outputs
