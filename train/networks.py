import torchvision as tv
import torch
import pytorch_lightning as ptl
import torch.functional as F
from collections import OrderedDict
import numpy as np

try:
    import dartinv
    from dartinv.train.data import CoupledDataset
    from dartinv.utils import BiDict
except ModuleNotFoundError:
    from train.data import CoupledDataset
    from utils import BiDict

from torch.utils.data import DataLoader


class SSSP(ptl.LightningModule):
    def __init__(self, model, tng_datasets, val_datasets, tst_datasets=None, discriminator=None, batch_size=32,
                 cons=None, spv_loss=None, gan_loss=None, optims=None, lr=1e-3, b1=0.99, b2=0.99, samex=False, 
                 samey=False, epoch_spec='max'):

        super(SSSP, self).__init__()
        self.model = model

        self._with_spv = spv_loss is not None
        self._with_gan = discriminator is not None and gan_loss is not None
        self._with_con = cons is not None

        if not self._with_spv and not self._with_gan and not self._with_con:
            raise Exception('You must supply a valid loss configuration.')

        # maps loss names to an identifier
        self.loss_config = BiDict()

        # maps loss names to the target net
        self.target_params = BiDict()

        self.tng_datasets = tng_datasets
        self.val_datasets = val_datasets
        self.tst_datasets = tst_datasets
        self.samex = samex
        self.samey = samey
        self.epoch_spec = epoch_spec

        # TODO: if tst_datasets is None cancel validation
        # TODO: add possibility to add transformers to Dataloaders

        ind = 0
        if self._with_spv:
            self.spv_loss = spv_loss

            self.loss_config['spv'] = ind
            self.target_params['spv'] = self.model.parameters()

            ind += 1

            # make sure we have corresponding datasets
            assert self._is_valid_spv_dsets()

        # make sure that if spv loss is called it precedes con loss (reuse y_pred)
        if self._with_con:
            self.cons = cons

            self.loss_config['con'] = ind
            self.target_params['con'] = self.model.parameters()

            ind += 1

            # make sure we have corresponding datasets
            assert self._is_valid_con_dsets()

        # make sure that gen loss call precedes dis loss (reuse yn_pred)
        if self._with_gan:
            self.discriminator = discriminator
            self.gan_loss = gan_loss

            self.loss_config['gen'] = ind
            self.target_params['gen'] = self.model.parameters()

            ind += 1

            self.loss_config['dis'] = ind
            self.target_params['dis'] = self.discriminator.parameters()
            ind += 1

            # make sure we have corresponding datasets
            assert self._is_valid_gan_dsets()

        # set rest of args
        self.batch_size = batch_size

        self.optims = optims
        self._lr = lr
        self._b1 = b1
        self._b2 = b2

        # instantiate in init for clarity
        self.y_pred = None
        self.yn_pred = None

    def _is_valid_spv_dsets(self):
        return 'xy' in self.tng_datasets and 'xy' in self.val_datasets

    def _is_valid_con_dsets(self):
        valid_samex = lambda dic : self.samex and 'xy' in dic
        valid_tng = 'xn' in self.tng_datasets or valid_samex(self.tng_datasets)
        valid_val = 'yn' in self.val_datasets or valid_samex(self.val_datasets)

        return valid_tng and valid_val

    def _is_valid_gan_dsets(self):
        valid_samex = lambda dic : self.samex and 'xy' in dic
        valid_samey = lambda dic : self.samey and 'xy' in dic

        valid_tng = ('xn' in self.tng_datasets or valid_samex(self.tng_datasets)) and \
                    ('yn' in self.tng_datasets or valid_samey(self.tng_datasets))

        valid_val = ('xn' in self.val_datasets or valid_samex(self.val_datasets)) and \
                    ('yn' in self.val_datasets or valid_samey(self.val_datasets))

        return valid_tng and valid_val

    def forward(self, x):
        return self.model(x)

    def con_loss(self, x, y_pred):
        return torch.Tensor([constraint(x=x, y=y_pred) for constraint in self.cons]).sum()

    def _get_loss_name(self, idx):
        return self.loss_config.inverse[idx][0]

    def _part_loss(self, batch_idx, loss_idx, x=None, y=None, xn=None, yn=None):
        # TODO: have different sized epochs

        # + + + + + + + + + +
        # CALCULATE  SPV LOSS
        # + + + + + + + + + +
        if 'spv' == self._get_loss_name(loss_idx):

            if x is None:
                self.y_pred = None
                return torch.Tensor([0]).requires_grad_()

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                x = x.cuda(x.device.index)
                y = y.cuda(y.device.index)

            self.y_pred = self.forward(x)
            loss = self.spv_loss(self.y_pred, y)

        # + + + + + + + + + +
        # CALCULATE  CON LOSS
        # + + + + + + + + + +
        if 'con' == self._get_loss_name(loss_idx):
            # TODO : make sure we really don't need any detach

            # if self.y_pred is None, no spv loss in configuration, train with unrelated dataset
            if self.y_pred is None:

                if xn is None:
                    return torch.Tensor([0]).requires_grad_()

                if self.on_gpu:
                    xn = xn.cuda(xn.device.index)

                y_pred = self.forward(xn)
            else:
                y_pred = self.y_pred

            loss = self.con_loss(x, y_pred)

        # + + + + + + + + + + + + +
        # CALCULATE  GAN LOSS : GEN
        # + + + + + + + + + + + + +
        if 'gen' == self._get_loss_name(loss_idx):

            if xn is None:
                return torch.Tensor([0]).requires_grad_()

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                xn = xn.cuda(xn.device.index)

            # generate images
            self.yn_pred = self.forward(xn)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            # TODO : mixing of true and false examples
            valid = torch.ones(xn.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(self.yn_pred.device.index)

            # adversarial loss is binary cross-entropy
            loss = self.gan_loss(self.discriminator(self.yn_pred), valid)

        # + + + + + + + + + + + + +
        # CALCULATE  GAN LOSS : DIS
        # + + + + + + + + + + + + +
        if 'dis' == self._get_loss_name(loss_idx):

            if yn is None:
                return torch.Tensor([0]).requires_grad_()

            valid = torch.ones(yn.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(yn.device.index)

            real_loss = self.gan_loss(self.discriminator(yn), valid)

            # how well can it label as fake?
            # TODO : mixing of true and false examples
            fake = torch.zeros(yn.size(0), 1)
            if self.on_gpu:
                fake = fake.cuda(yn.device.index)

            fake_loss = self.gan_loss(self.discriminator(self.yn_pred.detach()), fake)

            # discriminator loss is the average of these
            loss = (real_loss + fake_loss) / 2

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # TODO: better to separate spv and gan behaviour, move out
        # TODO: make sure we don't throw away any part of the batch, it looks ok in the training loop, but check
        # TODO: check whether optimizer_step is None is possible or whether it just puts 0
        # if there is only one loss, the training loop calls w/o optimizer_idx
        # in this case there is only one value in self.loss_config

        part_loss = self._part_loss(batch_idx, optimizer_idx, **batch)

        # make sure there is proper logging in case the supplied batch was None (due to unequally sized
        # epochs of different loss parts)
        if part_loss == 0:
            log = {}
        else:
            loss_name = self._get_loss_name(optimizer_idx)
            tqdm_dict = {loss_name + '_loss': part_loss, 'batch_idx': batch_idx}
            log = {'progress_bar': tqdm_dict,
                   'log': tqdm_dict}

        output = OrderedDict({'loss': part_loss})
        output.update(log)
        return output

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        # TODO: better to separate spv and gan behaviour, move out
        # TODO: make sure we don't throw away any part of the batch, it looks ok in the training loop, but check
        # TODO: check whether optimizer_step is None is possible or whether it just puts 0

        part_loss = self._part_loss(batch_idx, dataset_idx, **batch)

        # make sure there is proper logging in case the supplied batch was None (due to unequally sized
        # epochs of different loss parts)
        # make sure there is proper logging in case the supplied batch was None (due to unequally sized
        # epochs of different loss parts)
        if part_loss == 0:
            log = {}
        else:
            loss_name = self._get_loss_name(dataset_idx)
            tqdm_dict = {loss_name + '_val_loss': part_loss, 'batch_idx': batch_idx}
            log = {'progress_bar': tqdm_dict,
                   'log': tqdm_dict}

        output = OrderedDict({'val_loss': part_loss})
        output.update(log)
        return output

    def validation_end(self, outputs):
        return {'val_loss': 1}
        avg_loss = torch.stack([x for x in outputs['val_loss']]).mean()
        return avg_loss

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):

    def configure_optimizers(self):
        ordered_losses = sorted(list(self.loss_config.items()), key=lambda t: t[1])

        optims = []
        for name, idx in ordered_losses:

            if self.optims is not None and name in self.optims.keys():
                optims.append(optims[name])

            # define default optimizer to be Adam
            else:
                optims.append(torch.optim.Adam(self.target_params[name], lr=self._lr, betas=(self._b1, self._b2)))

        return optims, []

    @ptl.data_loader
    def train_dataloader(self):
        return DataLoader(CoupledDataset(self.tng_datasets, epoch_spec=self.epoch_spec,
                                         samex=self.samex, samey=self.samey),
                          batch_size=self.batch_size, collate_fn=CoupledDataset.collate)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(CoupledDataset(self.val_datasets, epoch_spec=self.epoch_spec,
                                         samex=self.samex, samey=self.samey),
                          batch_size=self.batch_size, collate_fn=CoupledDataset.collate)

    # @ptl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(CoupledDataset(self.tst_datasets, epoch_spec=self.epoch_spec, samex=self.samex, samey=self.samey),
    #                       batch_size=self.batch_size)

    # def on_epoch_end(self):
    #     z = torch.randn(8, self.hparams.latent_dim)
    #     # match gpu device (or keep as cpu)
    #     if self.on_gpu:
    #         z = z.cuda(self.last_imgs.device.index)
    #
    #     # log sampled images
    #     sample_imgs = self.forward(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

