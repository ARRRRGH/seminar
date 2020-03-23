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

    def __init__(self, channels, input_shape, hidden_size, embed_size, in_channels, reduce_kernel_size=3,
                 drop_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = EncoderCNN(channels, embed_size=embed_size, in_channels=in_channels,
                                  shape=input_shape, drop_rate=drop_rate)
        self.decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, drop_rate=drop_rate)

        self.reduce = nn.Conv1d(in_channels=self.seq_len, out_channels=1, kernel_size=reduce_kernel_size,
                                padding=reduce_kernel_size // 2)
        self.linear_head = nn.Linear(in_features=hidden_size, out_features=len(self._classes))

        self.loss = nn.CrossEntropyLoss()
        self.logsoft = nn.LogSoftmax()

    def forward(self, x):
        encs = []
        for t in range(x.shape[1]):
            encs.append(self.encoder(x[:, t, :, :, :]))
        out_hidden_states = self.decoder(torch.stack(encs, dim=1))

        return self.linear_head(torch.sigmoid(self.reduce(out_hidden_states).squeeze(1)))

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
        target = self.classes(target)

        # loss = self.loss(input=self.forward(data), target=self.classes(target))
        nll = - self.logsoft(self.forward(data))
        loss = nll[torch.arange(len(target)), target]

        # calc contingency table
        with torch.no_grad():

            pred = nll.argmin(dim=1)
            classes, counts = torch.unique(target, return_counts=True)

            tp_per_cls = {}
            fp_per_cls = {}
            fn_per_cls = {}
            tn_per_cls = {}
            for cls, pred_p_cls in zip(classes, counts):
                inds = torch.where(target == cls)[0]

                p_cls = len(inds)
                tp_per_cls[cls] = (pred[inds] == target[inds]).to(torch.int).sum()
                fp_per_cls[cls] = pred_p_cls - tp_per_cls[cls]
                fn_per_cls[cls] = p_cls - tp_per_cls[cls]
                tn_per_cls[cls] = target.shape[0] - p_cls - fn_per_cls[cls]

        tqdm_dict = {'val_loss': loss, 'batch_idx': batch_idx}
        log = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        output = OrderedDict({'val_loss': loss})
        output.update(log)

        output.update({'tp_per_cls': tp_per_cls})
        output.update({'fn_per_cls': fn_per_cls})
        output.update({'fp_per_cls': fp_per_cls})
        output.update({'tn_per_cls': tn_per_cls})

        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        std_loss = torch.stack([x['val_loss'] for x in outputs]).std()

        tp_per_cls = self._sum_metric_per_cls('tp_per_cls', outputs)
        fp_per_cls = self._sum_metric_per_cls('fp_per_cls', outputs)
        fn_per_cls = self._sum_metric_per_cls('fn_per_cls', outputs)
        # tn_per_cls = self._sum_metric_per_cls('tn', 'tn_per_cls', outputs)

        recall_per_cls = {'recall_' + str(cls_out): tp_per_cls[cls_in] / (tp_per_cls[cls_in] + fn_per_cls[cls_in] + 1e-7)
                          for cls_out, cls_in in self._classes.items()}

        precision_per_cls = {'precision_' + str(cls_out): tp_per_cls[cls_in] / (tp_per_cls[cls_in] + fp_per_cls[cls_in] + 1e-7)
                             for cls_out, cls_in in self._classes.items()}

        f1_per_cls = {'f1_' + str(cls_out): 2 * tp_per_cls[cls_in] /
                                            (2 * tp_per_cls[cls_in] + fn_per_cls[cls_in] + fp_per_cls[cls_in] + 1e-7)
                      for cls_out, cls_in in self._classes.items()}

        threat_sc_per_cls = {'threat_sc_' + str(cls_out): tp_per_cls[cls_in] /
                                                   (tp_per_cls[cls_in] + fn_per_cls[cls_in] + fp_per_cls[cls_in] + 1e-7)
                             for cls_out, cls_in in self._classes.items()}

        mean_recall = torch.stack(list(recall_per_cls.values())).mean()
        mean_precision = torch.stack(list(precision_per_cls.values())).mean()
        mean_f1 = torch.stack(list(f1_per_cls.values())).mean()
        mean_threat_sc = torch.stack(list(threat_sc_per_cls.values())).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'std_loss': std_loss}
        ret = {'val_loss': avg_loss, 'log': tensorboard_logs}
        ret.update(recall_per_cls)
        ret.update(precision_per_cls)
        ret.update(f1_per_cls)
        ret.update(threat_sc_per_cls)

        ret['mean_recall'] = mean_recall
        ret['mean_precision'] = mean_precision
        ret['mean_f1'] = mean_f1
        ret['mean_threat_sc'] = mean_threat_sc

        return ret

    def _sum_metric_per_cls(self, name_in, outputs):
        ret = {}
        for cls_out, cls_in in self._classes.items():
            try:
                sum = torch.stack([x[name_in][cls_in] for x in outputs
                             if cls_in in x['tp_per_cls']]).sum()

            except RuntimeError:  # catch possibility of no values
                sum = torch.Tensor([0])

            ret[cls_in] = sum
        return ret


class EncoderCNN(nn.Module):
    def __init__(self, channels, embed_size=1024, in_channels=8, shape=None, drop_rate=0.5):
        super().__init__()

        # cast to right dimensions
        # self.pre1 = nn.Linear(in_features=in_channels, out_features=in_channels // 2)
        # self.pre2 = nn.Linear(in_features=in_channels // 2, out_features=3)

        # self.upsample = nn.Upsample(scale_factor=50, mode='bilinear', align_corners=True)

        # self.convtr1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels*6)

        # get the pretrained densenet model
        # self.densenet = models.densenet121(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        # self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_ncl = in_channels
        for i, ncl in enumerate(channels):
            self.convs.append(nn.Conv2d(in_channels=last_ncl,
                                        out_channels=ncl, kernel_size=3, padding=1))

            self.bns.append(nn.BatchNorm2d(num_features=ncl))

            last_ncl = ncl

        # add another fully connected layer
        self.embed = nn.Linear(in_features=shape[1] * shape[2] * channels[-1], out_features=embed_size)
        # self.embed = nn.Linear(in_features=1000, out_features=embed_size)

        # dropout layer
        self.dropout = nn.Dropout2d(p=drop_rate)

        # activation layers
        # self.prelu = nn.PReLU()

    def forward(self, images):

        # preproc_images = self.upsample(self.pre2(torch.sigmoid(self.pre1(
        #                                 images.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2))
        # get the embeddings from the densenet
        # outs = self.dropout(self.prelu(self.densenet(preproc_images)))

        outs = images
        for bn, conv in zip(self.bns, self.convs):
            outs = torch.relu(bn(conv(outs)))

        # outs = self.pre1(outs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outs = self.dropout(outs).flatten(1)

        # pass through the fully connected
        embeddings = self.embed(outs)

        return embeddings


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, drop_rate=0.5):
        super().__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.hidden_size)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, embedded):

        # batch size
        batch_size = embedded.size(0)

        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(DEVICE)

        outputs = []
        # pass the caption word by word
        for t in range(embedded.size(1)):

            hidden_state, cell_state = self.lstm_cell(embedded[:, t, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.dropout(self.fc_out(hidden_state))

            # build the output tensor
            outputs.append(out)

        return torch.stack(outputs, dim=1)
