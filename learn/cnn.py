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

import networkx as nx
import itertools as it

try:
    from .cnn_lstm.convlstm import ConvLSTM
    from utils import BiDict
except ModuleNotFoundError:
    from seminar.learn.cnn_lstm.convlstm import ConvLSTM
    from seminar.utils import BiDict

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


class _BASE(ptl.LightningModule):
    def __init__(self, train_image_folder, val_image_folder, n_jobs, batch_size, hierarchy_weights,
                 seq_len=None, lr=1e-3, epoch_size=None, reader=None, stratified_input=False):
        super().__init__()

        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder

        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.n_jobs = n_jobs
        self.lr = lr
        self.hierarchy_weights = hierarchy_weights

        self.reader = reader
        self.stratified_input = stratified_input

        # create merged class map
        self._classes = BiDict(dict(self.val_dataloader().dataset.class_to_idx,
                                    **self.train_dataloader().dataset.class_to_idx))
        self._hierarchy_graph = self._construct_class_hierarchy_graph()
        self.max_considered_depth = self.hierarchy_weights.shape[1]

        self.clsin2clsout = lambda l: np.array(list(it.chain(*map(self._classes.inverse.get, np.array(l)))))

        self.seq_len = seq_len

        self.loss = nn.CrossEntropyLoss()
        self.logsoft = nn.LogSoftmax(dim=1)

    def _construct_int_classes(self):
        val_idx = self.val_dataloader().class_to_idx
        trn_idx = self.train_dataloader().class_to_idx

    def _construct_class_hierarchy_graph(self):
        graph = nx.Graph()
        graph.add_node('0')
        classes = list(map(str, self._classes.keys()))
        self._recurse_graph_levels('0', graph, 0, list(classes))
        return graph

    def _recurse_graph_levels(self, parent_node, graph, new_layer, subset_classes):
        new_nodes = set([str(cls_out)[new_layer] for cls_out in subset_classes if new_layer < len(str(cls_out))])

        if new_nodes != set():
            for new_node in new_nodes:
                new_subset = list(filter(lambda x: x.startswith(parent_node[1:] + new_node), subset_classes))
                self._recurse_graph_levels(parent_node + new_node, graph, new_layer + 1, new_subset)
                graph.add_node(parent_node + new_node)
                graph.add_edge(parent_node, parent_node + new_node)

    def _get_dataloader(self, path, is_val=False):
        is_valid_file = lambda path: path.endswith('npy')

        if self.reader is None:
            loader = lambda path: torch.Tensor(np.load(path))
        else:
            loader = lambda path: torch.Tensor(self.reader(path))

        dset = ImageFolder(path, loader=loader, is_valid_file=is_valid_file)

        if self.stratified_input and not is_val:
            weights = self._make_weights_for_balanced_classes(dset.imgs, len(dset.classes))
            weights = torch.DoubleTensor(weights)
            weights /= weights.sum()

            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            shuffle = False

        else:
            sampler = None
            shuffle = True

        if self.epoch_size is not None:
            dset = Subset(dset, np.random.choice(len(dset), self.epoch_size, replace=False))

        return DataLoader(dset, batch_size=self.batch_size, num_workers=self.n_jobs, shuffle=shuffle, sampler=sampler)

    def _make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1

        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            if count[i] > 0:
                weight_per_class[i] = N / float(count[i])
            else:
                weight_per_class[i] = 0

        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]

        return weight

    def train_dataloader(self):
        return self._get_dataloader(self.train_image_folder, is_val=False)

    def val_dataloader(self):
        return self._get_dataloader(self.val_image_folder, is_val=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def validation_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'] for x in outputs]).mean()
        std_loss = torch.cat([x['val_loss'] for x in outputs]).std()

        tensorboard_logs = {'val_loss': avg_loss, 'std_loss': std_loss}

        # calculate binary metrics
        for depth in range(1, self.max_considered_depth + 1):

            classes = set([co[:depth] for co, ci in self._classes.items()])

            tp_per_cls = self._sum_metric_per_cls('tp_per_cls_%d' % depth, outputs, classes)
            fp_per_cls = self._sum_metric_per_cls('fp_per_cls_%d' % depth, outputs, classes)
            fn_per_cls = self._sum_metric_per_cls('fn_per_cls_%d' % depth, outputs, classes)
            # tn_per_cls = self._sum_metric_per_cls('tn', 'tn_per_cls', outputs)

            # print(depth, tp_per_cls, classes)

            recall_per_cls = {'recall/d%d/' % depth + str(co): tp_per_cls[co] / (tp_per_cls[co] + fn_per_cls[co])
                              if tp_per_cls[co] + fn_per_cls[co] != 0
                              else np.nan
                              for co in classes}

            precision_per_cls = {'precision/d%d/' % depth + str(co): tp_per_cls[co] / (tp_per_cls[co] + fp_per_cls[co])
                                 if tp_per_cls[co] + fp_per_cls[co] != 0
                                 else np.nan
                                 for co in classes}

            f1_per_cls = {'f1/d%d/' % depth + str(co): 2 * tp_per_cls[co] / (2 * tp_per_cls[co] +
                                                                          fn_per_cls[co] + fp_per_cls[co])
                          if 2 * tp_per_cls[co] + fn_per_cls[co] + fp_per_cls[co] != 0
                          else np.nan
                          for co in classes}

            threat_sc_per_cls = {'threat_sc/d%d/' % depth + str(co): tp_per_cls[co] / (tp_per_cls[co] +
                                                                                    fn_per_cls[co] + fp_per_cls[co])
                                 if tp_per_cls[co] + fn_per_cls[co] + fp_per_cls[co] != 0
                                 else np.nan
                                 for co in classes}

            tensorboard_logs.update(recall_per_cls)
            tensorboard_logs.update(precision_per_cls)
            tensorboard_logs.update(f1_per_cls)
            tensorboard_logs.update(threat_sc_per_cls)

            # mean_recall = np.nanmean(list(recall_per_cls.values()))
            # mean_precision = np.nanmean(list(precision_per_cls.values()))
            # mean_f1 = np.nanmean(list(f1_per_cls.values()))
            # mean_threat_sc = np.nanmean(list(threat_sc_per_cls.values()))
            #
            # tensorboard_logs['mean_recall/d%d/' % depth] = mean_recall
            # tensorboard_logs['mean_precision/d%d/' % depth] = mean_precision
            # tensorboard_logs['mean_f1/d%d/' % depth] = mean_f1
            # tensorboard_logs['mean_threat_sc/d%d/' % depth] = mean_threat_sc

        ret = {'val_loss': avg_loss, 'log': tensorboard_logs}

        return ret

    def _calculate_contingency_table(self, pred, target):
        output = {}

        def contingency(pred, target, all_classes):
            # calc contingency table
            with torch.no_grad():
                cls = dict(zip(*np.unique(pred, return_counts=True)))
                classes = OrderedDict([(c, cls[c]) if c in cls else (c, 0) for c in all_classes])

                tp_per_cls = {}
                fp_per_cls = {}
                fn_per_cls = {}
                tn_per_cls = {}

                for cls, pred_p_cls in classes.items():
                    inds = np.where(target == cls)[0]
                    p_cls = len(inds)
                    # print(cls, inds, pred.shape, target.shape, target, pred)
                    tp_per_cls[cls] = (pred[inds] == target[inds]).sum()
                    fp_per_cls[cls] = pred_p_cls - tp_per_cls[cls]
                    fn_per_cls[cls] = p_cls - tp_per_cls[cls]
                    tn_per_cls[cls] = target.shape[0] - p_cls - fn_per_cls[cls]

            return tp_per_cls, fp_per_cls, tn_per_cls, fn_per_cls

        for depth in range(1, self.max_considered_depth + 1):
            tp_per_cls, fp_per_cls, tn_per_cls, fn_per_cls = contingency(np.array([p[:depth] for p in pred]),
                                                                         np.array([t[:depth] for t in target]),
                                                                         set([c[:depth] for c in self._classes.keys()]))
            output.update({'tp_per_cls_%d' % depth: tp_per_cls})
            output.update({'fn_per_cls_%d' % depth: fn_per_cls})
            output.update({'fp_per_cls_%d' % depth: fp_per_cls})
            output.update({'tn_per_cls_%d' % depth: tn_per_cls})

        return output

    def _sum_metric_per_cls(self, name_in, outputs, classes):
        ret = {}
        for cls_out in classes:
            lis = [x[name_in][cls_out] for x in outputs if cls_out in x[name_in]]
            ret[cls_out] = np.sum(lis, dtype=np.int16)
        return ret

    def training_step(self, batch, batch_idx):
        # data, target = batch
        # loss = self.loss(input=self.forward(data), target=target)
        # self.train()

        loss, _, _ = self.hierarchical_cross_entropy_loss(batch)
        loss = loss.mean()

        tqdm_dict = {'training_loss': loss, 'batch_idx': batch_idx}
        log = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        output = OrderedDict({'loss': loss})
        output.update(log)
        return output

    def validation_step(self, batch, batch_idx):
        # # loss = self.loss(input=self.forward(data), target=self.clsout2clsin(target))
        # nll = - self.logsoft(self.forward(data))
        # loss = nll[torch.arange(len(target)), target]
        # self.train()

        loss, pred, target = self.hierarchical_cross_entropy_loss(batch)
        pred_in, pred_out = pred
        target_in, target_out = target

        # print(pred, target)

        tqdm_dict = {'val_loss': loss, 'batch_idx': batch_idx}
        log = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        output = OrderedDict({'val_loss': loss})
        output.update(log)

        output.update(self._calculate_contingency_table(pred_out, target_out))
        return output

    def get_loss_and_pred(self, batch):
        data, in_targets = batch

        nlls = - self.logsoft(self.forward(data))
        preds_in = torch.argmin(nlls, dim=1)

        # if is not lstm, duplicate true labels
        if data.shape[0] != preds_in.shape[0]:
           in_targets = torch.cat([torch.Tensor([target] * self.seq_len) for target in in_targets]).long()

        reg = torch.zeros(len(preds_in))
        if hasattr(self, 'reg_invariance') and self.reg_invariance:
            for i in range(len(preds_in) // self.batch_size - 1):
                reg[i * self.batch_size: (i+1) * self.batch_size] = self.reg_invariance * \
                                                                    (len(torch.unique(preds_in)) - 1) / self.batch_size

        loss = nlls[torch.arange(len(in_targets)), in_targets] + reg
        return loss, preds_in

    def get_weights(self, preds_out, out_targets):
        # get shortest path
        with torch.no_grad():
            weights = []
            for pred, target in zip(preds_out, out_targets):
                dist = nx.shortest_path_length(self._hierarchy_graph,
                                               '0' + pred,
                                               '0' + target)

            layer_dist = max(0, self.max_considered_depth - dist // 2 - 1)
            layer0_cls = int(str(target)[0]) - 1
            weights.append(self.hierarchy_weights[layer0_cls, layer_dist])

        return weights

    def hierarchical_cross_entropy_loss(self, batch):
        data, in_targets = batch
        out_targets = self.clsin2clsout(in_targets)

        loss, preds_in = self.get_loss_and_pred(batch)
        preds_out = self.clsin2clsout(preds_in)

        weights = self.get_weights(preds_out, out_targets)
        return loss * torch.Tensor(weights).requires_grad_(False), (preds_in, preds_out), (in_targets, out_targets)
        # return loss, (preds_in, preds_out), (in_targets, out_targets)


class CNN(_BASE):
    def __init__(self, channels, input_shape, in_channels, kernel_size, linspace=1024, drop_rate=0.5, bn_momentum=0.5,
                 bn_track_running_stats=True, reg_invariance=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reg_invariance = reg_invariance
        self.encoder = EncoderCNN(channels=channels, embed_size=linspace, in_channels=in_channels, shape=input_shape,
                                  drop_rate=drop_rate, kernel_size=kernel_size, bn_momentum=bn_momentum,
                                  bn_track_running_stats=bn_track_running_stats)

        self.to_class = nn.Linear(in_features=linspace, out_features=len(self._classes))

    def forward(self, x):
        encs = []
        for t in range(x.shape[1]):
            encs.append(self.encoder(x[:, t, :, :, :]))

        rets = []
        for t in encs:
            rets.append(torch.sigmoid(self.to_class(t)))

        # now stack in batch dimension
        rets = torch.cat(rets, dim=0)
        print(rets.shape)
        return rets


class LSTM(_BASE):

    def __init__(self, input_shape, in_channels, hidden_channels, kernel_size, num_layers, bias=True, reduce_channels=5,
                 reduce_hidden=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lstm = ConvLSTM(in_channels, hidden_channels, kernel_size, num_layers, batch_first=True, bias=bias,
                             return_all_layers=False)

        padding = kernel_size[-1][0] // 2, kernel_size[-1][1] // 2
        #self.conv = nn.Conv2d(in_channels=hidden_channels[-1], kernel_size=kernel_size[-1], out_channels=1,
        #                      padding=padding)

        hidden_size = hidden_channels[-1] * int(np.prod(input_shape))

        self.linear_head = nn.Linear(in_features=hidden_size * reduce_channels, out_features=len(self._classes))
        self.reduce = nn.Conv2d(in_channels=1, out_channels=reduce_channels, kernel_size=(reduce_hidden, self.seq_len),
                                stride=1, padding=(reduce_hidden // 2, 0))

    def forward(self, x):
        h, c = self.lstm(x)
        h, c = h[0], c[0]

        output = []
        for t in range(h.shape[1]):
            output.append(h[:, t, :, :, :].flatten(1))
        out_hidden_states = torch.stack(output, dim=1)

        #print(out_hidden_states.shape)
        red = self.reduce(out_hidden_states.permute(0, 2, 1).unsqueeze(1)).squeeze(-1).flatten(1)
        #print(red.shape)
        return self.linear_head(torch.sigmoid(red))


class LSTM2(_BASE):

    def __init__(self, channels, input_shape, hidden_size, embed_size, in_channels, reduce_kernel_size=3, drop_rate=0.5,
                 bn_momentum=0.5, bn_track_running_stats=True, kernel_size=5, reduce_hidden=2,  reduce_groups=1, reduce_step=1, reduce_channels=5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = EncoderCNN(channels, embed_size=embed_size, in_channels=in_channels, shape=input_shape,
                                  drop_rate=drop_rate, bn_momentum=bn_momentum, kernel_size=kernel_size,
                                  bn_track_running_stats=bn_track_running_stats)

        self.decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, drop_rate=drop_rate)

        #self.reduce = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // reduce_hidden, groups=reduce_groups, kernel_size=reduce_kernel_size,
        #                        padding=reduce_kernel_size // 2)
        #self.reduce2 = nn.Linear(in_features=self.seq_len, out_features=1)
        self.reduce = nn.Conv2d(in_channels=1, out_channels=reduce_channels, kernel_size=(reduce_hidden, self.seq_len), stride=1, padding=(reduce_hidden // 2, 0))
        self.linear_head = nn.Linear(in_features=hidden_size * reduce_channels, out_features=len(self._classes))


    def forward(self, x):
        encs = []
        for t in range(x.shape[1]):
            encs.append(self.encoder(x[:, t, :, :, :]))
        out_hidden_states = torch.relu(self.decoder(torch.stack(encs, dim=1)))

        #red = torch.relu(self.reduce(out_hidden_states.permute(0, 2, 1))).squeeze(-1)
        #red = self.reduce2(red).squeeze(-1)

        red = self.reduce(out_hidden_states.permute(0, 2, 1).unsqueeze(1)).squeeze(-1).flatten(1)
        return self.linear_head(torch.sigmoid(red))


class EncoderDense(nn.Module):
    def __init__(self, channels, embed_size=1024, in_channels=8, shape=None, drop_rate=0.5):
        super().__init__()

        # cast to right dimensions
        # self.pre1 = nn.Linear(in_features=in_channels, out_features=in_channels // 2)
        # self.pre2 = nn.Linear(in_features=in_channels // 2, out_features=3)

        self.upsample = nn.Upsample(scale_factor=50, mode='*bilinear', align_corners=True)
        # self.convtr1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels*6)

        # get the pretrained densenet model
        self.densenet = models.densenet121(pretrained=True)

        # replace the classifier with a fully connected embedding layer
        # self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)

        # add another fully connected layer
        self.embed = nn.Linear(in_features=1000, out_features=embed_size)
        # self.embed = nn.Linear(in_features=1000, out_features=embed_size)

        # dropout layer
        self.dropout = nn.Dropout2d(p=drop_rate)

        # activation layers  # self.prelu = nn.PReLU()

    def forward(self, images):
        preproc_images = self.upsample(
            self.pre2(torch.sigmoid(self.pre1(images.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2))
        # get the embeddings from the densenet
        outs = self.dropout(self.prelu(self.densenet(preproc_images)))
        outs = self.dropout(outs).flatten(1)

        # pass through the fully connected
        embeddings = self.embed(outs)

        return embeddings


class EncoderCNN(nn.Module):
    def __init__(self, channels, embed_size=1024, in_channels=8, shape=None, drop_rate=0.5, bn_momentum=0.5,
                 bn_track_running_stats=True, kernel_size=5):
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
            self.bns.append(nn.BatchNorm2d(num_features=last_ncl, momentum=bn_momentum,
                                           track_running_stats=bn_track_running_stats))

            self.convs.append(nn.Conv2d(in_channels=last_ncl, out_channels=ncl, kernel_size=kernel_size, padding=kernel_size//2))

            last_ncl = ncl

        # add another fully connected layer
        self.embed = nn.Linear(in_features=shape[1] * shape[2] * channels[-1], out_features=embed_size)
        # self.embed = nn.Linear(in_features=1000, out_features=embed_size)

        # dropout layer
        self.dropout = nn.Dropout2d(p=drop_rate)

        # activation layers  # self.prelu = nn.PReLU()

    def forward(self, images):

        # preproc_images = self.upsample(self.pre2(torch.sigmoid(self.pre1(
        #                                 images.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2))
        # get the embeddings from the densenet
        # outs = self.dropout(self.prelu(self.densenet(preproc_images)))

        outs = images
        for bn, conv in zip(self.bns, self.convs):
            outs = torch.relu(conv(bn(outs)))

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
