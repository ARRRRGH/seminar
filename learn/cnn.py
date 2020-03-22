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


use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


class _LSTM(ptl.LightningModule):
    def __init__(self, classes, train_image_folder, val_image_folder, n_jobs, batch_size, seq_len=None,
                 lr=1e-3, epoch_size=None):
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

        self.seq_len = seq_len

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


class LSTM(_LSTM):

    def __init__(self, input_shape, in_channels, hidden_channels, kernel_size, num_layers, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lstm = ConvLSTM(in_channels, hidden_channels, kernel_size, num_layers,
                             batch_first=True, bias=bias, return_all_layers=False)

        padding = kernel_size[-1][0] // 2, kernel_size[-1][1] // 2
        self.conv = nn.Conv2d(in_channels=hidden_channels[-1], kernel_size=kernel_size[-1], out_channels=1,
                              padding=padding)

        self.linear_head = nn.Linear(in_features=int(np.prod(input_shape)), out_features=len(self._classes))
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

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


class LSTM2(_LSTM):

    def __init__(self, hidden_size, embed_size, in_channels, reduce_kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = EncoderCNN(embed_size=embed_size, in_channels=in_channels)
        self.decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size)

        self.reduce = nn.Conv1d(in_channels=self.seq_len, out_channels=1, kernel_size=reduce_kernel_size)
        self.linear_head = nn.Linear(in_features=hidden_size, out_features=len(self._classes))

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        encs = []
        for t in range(x.shape[1]):
            encs.append(self.encoder(x[:, t, :, :, :]))

        out_hidden_states = self.decoder(torch.stack(encs, dim=1))
        # return self.linear_head(torch.max(out_hidden_states, dim=1)[0])
        return self.linear_head(self.reduce(out_hidden_states).squeeze(1))

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

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=1024, in_channels=8):
        super().__init__()

        assert in_channels > 6

        # cast to right dimensions
        self.pre1 = nn.Linear(in_features=in_channels, out_features=in_channels//2)
        self.pre2 = nn.Linear(in_features=in_channels//2, out_features=3)

        # get the pretrained densenet model
        #self.densenet = models.densenet121(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        #self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

        # add another fully connected layer
        self.embed = nn.Linear(in_features=1024, out_features=embed_size)

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # activation layers
        self.prelu = nn.PReLU()

    def forward(self, images):

        preproc_images = self.pre2(torch.sigmoid(self.pre1(images.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)

        # get the embeddings from the densenet
        # outs = self.dropout(self.prelu(self.densenet(preproc_images)))

        outs = self.prelu(self.conv3(self.sigmoid(self.conv2(self.sigmoid(self.conv1(preproc_images))))))
        outs = self.dropout(outs)
        # pass through the fully connected
        embeddings = self.embed(outs)

        return embeddings


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, out_classes=10):
        super().__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.out_classes = out_classes

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.hidden_size)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.out_classes)

    def forward(self, features, embedded):

        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)

        outputs = []
        # pass the caption word by word
        for t in range(embedded.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(embedded[:, t, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.fc_out(hidden_state)

            # build the output tensor
            outputs.append(out)

        return torch.stack(outputs, dim=1)
