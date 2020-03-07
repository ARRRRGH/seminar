from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class CoupledDataset(Dataset):
    _epoch_spec = dict([('max', max), ('min', min)])
    _valid_keys = ['xy', 'xn', 'yn']

    def __init__(self, datasets, epoch_spec='max', samex=False, samey=False):

        """

        :param datasets: dictionary with keys 'xy', 'xn', 'yn' mapping to GeneralDataset with overflow='modulo',
                         'xy' must be double, 'xn' and 'yn' must be single

        """

        super(CoupledDataset, self).__init__()

        self.datasets = datasets
        if not type(datasets) is dict:
            raise TypeError('datasets must be a dict.')

        # check that only valid keys are supplied
        for key in self.datasets:
            if not key in self._valid_keys:
                raise KeyError('You supplied invalid key: ' + str(key))

        self.epoch_spec = epoch_spec

        self.samex = samex
        self.samey = samey

        # for name in ['xy', 'xn', 'yn']:
        #     if name not in self.datasets:
        #         self.datasets[name] = VoidDataset()

    def __getitem__(self, index):
        # TODO: implement non-sequentially
        ret = {}

        if 'xy' in self.datasets:
            xy = self.datasets['xy'][index]
            ret['x'] = xy[0]
            ret['y'] = xy[1]

        if self.samex and 'xy' in self.datasets:
            ret['xn'] = ret['x']
        elif 'xn' in self.datasets:
            ret['xn'] = self.datasets['xn'][index]

        if self.samey and 'xy' in self.datasets:
            ret['yn'] = ret['y']
        elif 'yn' in self.datasets:
            ret['yn'] = self.datasets['yn'][index]

        return ret

    def __len__(self):
        return self._epoch_spec[self.epoch_spec]([len(ds) for ds in self.datasets.values()])

    @staticmethod
    def collate(batch):
        nones = dict([(k, None) for b in batch for k, v in b.items() if v is None])

        # This removes the edge case where a key might map to None only in some batches
        # TODO: don't wrap default_collate so this is not necessary.
        no_nones = [dict(filter(lambda t: t[0] not in nones.keys(), elem.items())) for elem in batch]

        batch = torch.utils.data._utils.collate.default_collate(no_nones)
        batch.update(nones)
        return batch


class VoidDataset(Dataset):
    def __getitem__(self, index):
        return None, None

    def __len__(self):
        return 0


class ImageDataset(Dataset):
    def __init__(self, root_csv, reader, reader2=None, overflow='modulo'):
        super(ImageDataset, self).__init__()
        self.paths = pd.read_csv(root_csv)
        self._overflow = overflow

        # data with ground truth
        if self.paths.shape[1] == 2:
            self._is_double = True
            self.reader1 = reader

            self.reader2 = reader
            if reader2 is not None:
                self.reader2 = reader2

        elif self.paths.shape[1] == 1:
            self._is_double = False
            self.reader1 = reader

        else:
            raise Exception('Provided csv has not the right format.')

    def __getitem__(self, index):
        # TODO: circumvent this somehow
        # if index > len(self) and self._overflow == 'modulo':
        #     index %= len(self)

        if not self._is_double:
            if index >= len(self):
                return None

            ret1 = self.reader1(self.paths.iloc[index, 0])
            return ret1
        else:
            if index >= len(self):
                return None, None

            ret1 = self.reader1(self.paths.iloc[index, 0])
            ret2 = self.reader2(self.paths.iloc[index, 1])
            return ret1, ret2

    def __len__(self):
        return len(self.paths)
