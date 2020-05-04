import xarray as xr
import numpy as np
import pandas as pd
from functools import partial
import copy
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score

try:
    from utils import run_jobs
    from preproc.transformers import InputList, TransformerSwitch
except ModuleNotFoundError:
    import seminar
    from seminar.utils import run_jobs
    from seminar.preproc.transformers import InputList, TransformerSwitch

from collections import OrderedDict


def pred_array(model, inp, arr=None, model_arr=None, n_batches=1000, no_val=-1, n_jobs=6, *args, **kwargs):
    assert arr is not None or model_arr is not None
    if arr is None:
        arr = np.ones(model_arr.shape) * no_val
        shape = model_arr.shape
    if model_arr is None:
        shape = arr.shape

    inp_is_list = type(inp) is InputList

    # run in multiple batches for memory

    if not inp_is_list:
        splits = [(spl.index, spl.values) for spl in np.array_split(inp, min(n_batches, len(inp)))]
        split = lambda nth_batch: splits[nth_batch]

        orig_index = inp.index
        n_batches = len(splits)
    else:
        split_model = np.array_split(inp.get(0), min(n_batches, len(inp.get(0))))

        def split(n):
            spl_model = split_model[n]
            inp_list = InputList(OrderedDict([(name, inp.get(name).loc[spl_model.index].values) for name in inp.names if inp.get(name) is not None]))
            return spl_model.index, inp_list

        orig_index = inp.get(0).index
        n_batches = len(split_model)

    def predict_batch(nth_batch):
        index, batch = split(nth_batch)
        return index, model.predict(batch)

    jobs = []
    for j in range(n_batches):
        jobs.append(partial(predict_batch, nth_batch=j))

    res = run_jobs(jobs, n_jobs=n_jobs, *args, **kwargs)

    out_df = pd.DataFrame(columns=['pred'], index=orig_index)
    out_df = out_df.fillna(no_val)

    for i, pred in res[0]:
        out_df.loc[i, 'pred'] = pred
        arr[np.unravel_index(i, shape)] = pred

    arr = xr.DataArray(arr, dims=model_arr.dims)
    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr, out_df


class InputListSplitter(object):
    def split(self, n_splits, X, slic=slice):
        step = len(X) // n_splits
        for i in range(n_splits):
            yield slic(i * step, (i+1) * step)

    def get_cvs(self, cv, X):
        splits = list(self.split(cv, X, slic=np.arange))

        cvs = []
        all_n = set(range(cv))
        for i in all_n:
            train = all_n - {i}
            train_inds = np.concatenate([splits[j] for j in train])
            test_inds = splits[i]

            cvs.append((train_inds, test_inds))

        return cvs

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits
    

class GridSearch(object):
    score_dict = {'adjusted_rand_score': adjusted_rand_score, 'adjusted_mutual_info_score': adjusted_mutual_info_score,
                  'fowlkes_mallows_score': fowlkes_mallows_score}

    def __init__(self, estimator, parameter_grid, cv=5, scores=None):
        self.estimator = estimator

        self.parameter_grid = OrderedDict(parameter_grid)

        self.splitter = InputListSplitter()
        self.cv = cv

        self.scores = scores
        if self.scores is None:
            self.scores = ['adjusted_rand_score']

        self.parameter_names = list(self.parameter_grid.keys())
        self.permuts = list(it.product(*[range(len(val)) for key, val in parameter_grid.items()]))
        self.param_dicts = [{self.parameter_names[ax]: self.parameter_grid[self.parameter_names[ax]][val]
                             for ax, val in enumerate(perm)} for perm in self.permuts ]

    def fit(self, X, ys, verbose=False, *args, **kwargs):

        if type(ys) is not list:
            ys = [ys]

        inds = self.splitter.get_cvs(self.cv, X)

        jobs = [partial(self._fit_and_eval, params=params, X=X, ys=ys, train=train, test=test, scores=self.scores,
                        estimator=copy.deepcopy(self.estimator), run_ident=i, param_ident=j, verbose=verbose)
                for i, (train, test) in enumerate(inds) for j, params in enumerate(self.param_dicts)]

        res = run_jobs(jobs, *args, **kwargs)[0]
        res = np.stack(res, axis=1)

        param_idents = [{int(pid): np.where(res[i, :, 0] == pid)[0] for pid in np.unique(res[i, :, 0])} for i in range(len(ys))]
        merged = [{pid: res[i, param_idents[i][pid], 2:].mean(axis=0) for pid in param_idents[i].keys()
                   if not np.any(np.isnan(res[i, param_idents[i][pid], 2:]))}
                  for i in range(len(ys))]

        params = {i: self.param_dicts[i].copy() for i in merged[0].keys()}

        return merged, params

    def _fit_and_eval(self, params, X, ys, run_ident, param_ident, train, test, scores=None, estimator=None, verbose=False):
        if estimator is None:
            estimator = self.estimator
        try:
            # remove non set key
            set_params = {}
            # empty_params = {}
            for key, val in params.items():
                if not val == '--':
                    set_params[key] = val
                #else:
                #    try:
                #        empty_params[key] = estimator.get_param(key)
                #    except:
                #        pass

            _ = estimator.set_params(**set_params)
        except (ValueError, AttributeError) as e:
            if verbose:
                print('Tried to set ', str(set_params), ' but failed. Details: ', str(e))

            return np.concatenate([np.array([[param_ident, run_ident] for i in range(len(ys))]),
                                   np.array([[np.nan] * len(scores) for i in range(len(ys))])], axis=1)

        xtrain, ystrain = copy.deepcopy(X[train]), [copy.deepcopy(y[train]) for y in ys]
        xtest, ystest = copy.deepcopy(X[test]), [copy.deepcopy(y[test]) for y in ys]

        has_predict = hasattr(estimator[-1], 'predict') and (hasattr(estimator[-1].get(), 'predict')
                                                             if type(estimator[-1]) is TransformerSwitch else True)
        if has_predict:
            _ = estimator.fit(xtrain)
            scores = np.array([[self.score_dict[score](ytest.flatten(), estimator.predict(xtest)) for score in scores]
                               for ytest in ystest])
        else:
            labels = estimator.fit_predict(xtrain)
            scores = np.array([[self.score_dict[score](ytrain.flatten(), labels) for score in scores] for ytrain in ystrain])

        return np.concatenate([np.array([[param_ident, run_ident] for i in range(len(ystest))]), scores], axis=1)

