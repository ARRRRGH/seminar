import numpy as np
import itertools

from scipy import signal
import scipy.fftpack as ff

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone as skl_clone
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from collections import OrderedDict


class InputList(object):
    def __init__(self, lis):
        if type(lis) is list:
            self.map_inp_name = {i: i for i in range(len(lis))}
            self.names = list(self.map_inp_name.keys())

            self.list = lis
            self.map = OrderedDict([(ident, df) for ident, df in enumerate(lis)])

        elif type(lis) is OrderedDict:
            self.map_inp_name = {inp_name: out_name for out_name, inp_name in enumerate(lis.keys())}
            self.names = list(self.map_inp_name.keys())

            self.list = list(lis.values())
            self.map = OrderedDict([(self.map_inp_name[inp_name], lis[inp_name]) for inp_name in lis.keys()])

            # add indices as keys so that get works with name and index,
            # these additional keys do not appear in self.names
            self.map_inp_name.update({i: i for i in range(len(lis))})

        self.map[None] = None

        self.mapper = np.vectorize(lambda inp_name: self.map_inp_name[inp_name])

    def get(self, *args, outtyp='list'):
        if outtyp == 'list':
            lis = [self.map.get(name, None) for name in self.mapper(args)]

            if len(lis) == 1:
                return lis[0]
            else:
                return lis

        elif outtyp == 'dict':
            return OrderedDict([(name, self.map.get(self.map_inp_name.get(name, None), None)) for name in args])

    def __len__(self):
        return len(self.list)


class TransformerSwitch(BaseEstimator, TransformerMixin):
    def __init__(self, is_on=0, transformers=None):
        super(TransformerSwitch, self).__init__()
        self.is_on = is_on
        self.transformers = transformers

        if self.transformers is None:
            self.transformers = [None]

    def fit(self, *args, **kwargs):
        transformer = self.transformers[self.is_on]
        if transformer is not None:
            return transformer.fit(*args, **kwargs)
        else:
            return self

    def transform(self, X, *args, **kwargs):
        transformer = self.transformers[self.is_on]
        if transformer is not None:
            return transformer.transform(X, *args, **kwargs)
        else:
            return X

    def predict(self, X, *args, **kwargs):
        return self.transformers[self.is_on].predict(X, *args, **kwargs)

    def get(self):
        return self.transformers[self.is_on]


class FFTfeatures(BaseEstimator, TransformerMixin):
    def __init__(self, thr_freq, scale=False, scaler=None, normalize_psd=True, quantile_range=(0.25, 0.75),
                 time_step=6 / 365, pre_feature_name=''):
        super().__init__()

        self.thr_freq = thr_freq
        self.scale = scale
        self.normalize_psd = normalize_psd
        self.quantile_range = quantile_range
        self.time_step = time_step
        self.scaler = scaler
        self.pre_feature_name = pre_feature_name

    def fit(self, X, *args, **kwargs):
        if self.scale:
            ret = self._fft(X, fit=True)
            if self.scaler is None:
                self.scaler = RobustScaler(self.quantile_range).fit(ret)  # self.scaler = StandardScaler().fit(ret)
            else:
                self.scaler = self.scaler.fit(ret)
        else:
            ret = self._fft(X[0:1, :], fit=True)
        return self

    def transform(self, X):
        return self._fft(X)

    def _fft(self, X, fit=False):
        # detrend
        detrended = signal.detrend(X, type='linear')

        # get power spectral density
        fft_psd = np.abs(ff.fft(detrended))

        if fit:
            self.freqs = ff.fftfreq(fft_psd.shape[1], self.time_step)
            self.valid_freq_mask = np.logical_and(self.thr_freq > self.freqs, self.freqs > 0)
            self.out_feat_dim = len(self.freqs[self.valid_freq_mask])

        fft_psd = fft_psd[..., self.valid_freq_mask]
        if self.normalize_psd:
            nrm = np.sum(fft_psd, axis=1)
            fft_psd = np.einsum('ij, i -> ij', fft_psd, 1 / (nrm + 1e-12))

        # check is same input/output type
        assert fft_psd.shape[1] == self.out_feat_dim

        if self.scale and hasattr(self, 'scaler'):
            fft_psd = self.scaler.transform(fft_psd)

        return fft_psd

    def get_feature_names(self):
        check_is_fitted(self, ['valid_freq_mask', 'freqs'])
        return [self.pre_feature_name + '%.3f' % freq for freq in self.freqs[self.valid_freq_mask]]


class _FixedCombo(object):
    def __init__(self, transformer, edge_cols=None, len_inp=None, is_concerned=None, clone=False, ordered_tkwargs=None,
                 *targs, **tkwargs):

        if edge_cols is not None:
            len_inp = len(edge_cols) + 1
        else:
            assert len_inp is not None

        if ordered_tkwargs is None:
            ordered_tkwargs = [None] * len_inp

        if not clone:
            self.transformers = [transformer(*targs, **dict(tkwargs, **ordered_tkwargs[i]))
                                 for i in range(len_inp)]

        else:
            self.transformers = [skl_clone(transformer) for i in range(len_inp)]

        self.edge_cols = edge_cols
        self.is_concerned = is_concerned
        if self.is_concerned is None:
            self.is_concerned = [True] * len_inp

    def fit(self, X, *args, **kwargs):
        if self.edge_cols is None:
            self._do_all('fit', spec_kwargs=[{'X': X.get(i)} for i in range(len(X))])
        else:
            self._do_all('fit', spec_kwargs=[{'X': x} for x in np.hsplit(X, self.edge_cols)])
        return self

    def transform(self, X):
        if self.edge_cols is None:
            ret = self._do_all('transform', spec_kwargs=[{'X': X.get(i)} for i in range(len(X))])
        else:
            ret = self._do_all('transform', spec_kwargs=[{'X': x} for x in np.hsplit(X, self.edge_cols)])
        return np.concatenate(ret, axis=1)

    def _do_all(self, method, spec_args=None, spec_kwargs=None, *args, **kwargs):
        spec_args, spec_kwargs = self._default_args_kwargs(spec_args, spec_kwargs, len(self.transformers))

        # TODO: do this in parallel
        ret = []
        for i, (sa, skw, t) in enumerate(zip(spec_args, spec_kwargs, self.transformers)):
            if self.is_concerned[i]:
                ret.append(getattr(t, method)(*sa, *args, **skw, **kwargs))

        return ret

    def _default_args_kwargs(self, args, kwargs, nr):
        if args is None:
            args = [()] * nr
        if kwargs is None:
            kwargs = [{}] * nr

        return args, kwargs

    def get_feature_names(self):
        return list(itertools.chain(*[t.get_feature_names() for i, t in enumerate(self.transformers) if self.is_concerned[i]]))

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

        for t in self.transformers:
            t.set_params(**kwargs)


class FFCombo(BaseEstimator, TransformerMixin, _FixedCombo):
    def __init__(self, thr_freq=30, edge_cols=None, len_inp=None, is_concerned=None, scale=False, normalize_psd=True,
                 quantile_range=(0.25, 0.75), scaler=None, pre_feature_name='', time_step=6/360, ordered_tkwargs=None):
        _FixedCombo.__init__(self, edge_cols=edge_cols, transformer=FFTfeatures,
                             len_inp=len_inp, is_concerned=is_concerned, thr_freq=thr_freq, scale=scale,
                             normalize_psd=normalize_psd, quantile_range=quantile_range, scaler=scaler,
                             pre_feature_name=pre_feature_name, time_step=time_step,
                             ordered_tkwargs=ordered_tkwargs)

        self.len_inp = len_inp
        self.ordered_tkwargs = ordered_tkwargs
        for attr in self.transformers[0]._get_param_names():
            setattr(self, attr, getattr(self.transformers[0], attr))


class FuncTransformerCombo(_FixedCombo):
    def __init__(self, transformer, len_inp=None, edge_cols=None, is_concerned=None, clone=True, *args, **kwargs):
        _FixedCombo.__init__(self, edge_cols=edge_cols, len_inp=len_inp, transformer=transformer, is_concerned=is_concerned,
                             clone=clone, *args, **kwargs)

        for attr in self.transformers[0]._get_param_names():
            setattr(self, attr, getattr(self.transformers[0], attr))
