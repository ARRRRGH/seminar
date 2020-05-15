import xarray as xr
import numpy as np
import pandas as pd
from functools import partial
import copy
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, v_measure_score
from sklearn.metrics import confusion_matrix
import itertools as it
from RSreader import io
import os

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
            inp_list = InputList(OrderedDict([(name, inp.get(name).loc[spl_model.index]) for name in inp.names if inp.get(name) is not None]))
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
                  'fowlkes_mallows_score': fowlkes_mallows_score, 'v_measure_score': v_measure_score}

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

        res = run_jobs(jobs, verbose=verbose, *args, **kwargs)[0]
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


def simulated_annealing(x0, energy, label_set, label_weights=None, epochs=1, T=20.0, eta=0.99995, max_change=3,
                        subset=1, verbose=False, sample=None):
    """Simulated Annealing for TSP

    T(n) = T * eta**n

    """
    np.random.seed(7)

    N = x0.shape[0]

    energies = []
    state_perm = x0
    perm = state_perm.copy()

    label_dict = {l: i for i, l in enumerate(label_set)}
    label_mapper = np.vectorize(lambda ind: label_dict.get(ind))

    energ = np.ones(len(label_set))

    if label_weights is None:
        label_weights = np.ones(len(label_set)) / len(label_set)

    labels, new_energ = energy(perm)
    label_inds = label_mapper(labels)

    energ[label_mapper(labels)] = new_energ
    energies.append((perm.copy(), energ.copy(), T))

    samples = []

    T_n = T
    for e in range(epochs):
        if verbose:
            print('\r\r Epoch %d, T=%.9f: \n perm %s, \n energy %s' % (e, T_n, str(state_perm), str(energies[-1][1].mean())))

        print({l: energ[i] for l, i in label_dict.items()})

        for idx in np.random.permutation(N):

            if max_change > 1:
                n_inds = np.random.choice(range(1, max_change + 1), 1)
                inds = np.r_[np.random.choice(np.setdiff1d(np.arange(N), np.array([idx])), n_inds - 1), idx]

            else:
                inds = [idx]

            if len(inds) ==  1:
                new_label = np.random.choice(np.setxor1d(label_set, perm[idx]), 1)[0]
            else:
                new_label = np.random.choice(label_set, 1)[0]

            labels, new_energ = energy(perm, inds=inds, new_label=new_label, subset=subset)
            label_inds = label_mapper(labels)

            diff = np.sum((new_energ - energ[label_inds]) * label_weights[label_inds]) / np.sum(label_weights[label_inds])
            p = min(1, np.exp(- (diff.max() / T_n)))
            b = np.random.binomial(1, p, 1)

            if b == 1:
                state_perm[inds] = new_label
                energ[label_inds] = new_energ
            else:
                pass

            if sample is not None and e > epochs - sample:
                samples.append(state_perm.copy())

            perm = state_perm.copy()
            energ = energ.copy()

        T_n *= eta

        energies.append((state_perm, energ, T_n))

    return np.asarray(state_perm), energies, T_n, samples


def contingency_distance(prev_perm, score, clustering, ground_truth, valid_pix=None, inds=None, new_label=None,
                         subset=1, trn_inds=None):

    assert not(valid_pix is not None and trn_inds is not None)
    indices = trn_inds
    if indices is None:
        indices = valid_pix

    if indices is None:
        indices = slice(None, None)

    if inds is None and new_label is None:
        prev_perm = dict(zip(range(len(prev_perm)), prev_perm))
        prev_perm[-1] = -1

        mapper = np.vectorize(lambda ind: prev_perm.get(ind))
        clf_perm = mapper(clustering[indices])

        affected_labels = np.sort(np.unique(ground_truth[indices]))
        return affected_labels, 1 - score(clf_perm, ground_truth[indices], average=None)

    if not hasattr(inds, '__len__'):
        inds = [inds]

    affected_cluster_to = np.where(prev_perm == new_label)[0]
    affected_cluster_fro = np.concatenate([np.where(prev_perm == prev_perm[i])[0] for i in inds])
    affected_clusters = np.unique(np.r_[affected_cluster_to, affected_cluster_fro])

    affected_labels = [prev_perm[i] for i in affected_clusters] + [new_label]
    affected_labels_set = np.sort(np.unique(affected_labels))

    perm = prev_perm.copy()
    perm[inds] = new_label
    perm = dict(zip(affected_clusters, perm[affected_clusters]))

    mapper = np.vectorize(lambda ind: perm.get(ind, -1))

    pred = clustering[indices]
    gtr = ground_truth[indices]

    # only calculate on a random subset
    if subset != 1:
        subset = np.random.choice(range(len(gtr)), int(subset * len(gtr)))

        pred = pred[subset]
        gtr = gtr[subset]

    pred = mapper(pred)

    scores = score(pred, gtr, labels=affected_labels_set) #labels=affected_labels_set)
    invscore = 1 - scores

    return affected_labels_set, invscore


def train_predict_conf(pip, dset, model_arr, ground_truth, classif_out_path, samples, gt=None, is_aligned=False, train=True,
                       load_classif=False, gt_nan=-1, pred_nan=-1, dtype=np.uint16, *args, **kwargs):

    if type(samples) is int:
        samples = dset.sample(min(samples, len(dset)))

    if train:
        # print('fit model ')
        pip.fit(samples, gt)

    if not load_classif:
        # print('Classify dset ')
        classif = pred_array(model=pip, inp=dset, model_arr=model_arr, no_val=-1, *args, **kwargs)[0]

        # print('Write out classification to ' + classif_out_path)
        classif = io.write_out(arr=classif.astype(dtype), default_meta=classif.attrs, dst_path=classif_out_path)
    else:
        classif, _ = io.read_raster(classif_out_path)

    # print('Align classification and ground truth, write out to ' + align_path)
    if not is_aligned:
        # path arithmetic
        base, name = os.path.split(classif_out_path)
        name, ext = os.path.splitext(name)
        align_path = os.path.join(base, name + '_aligned' + ext)

        classif = io.align(ground_truth, classif, align_path)

    return classif, validate(classif.data, ground_truth.data, pred_nan=pred_nan, gt_nan=gt_nan)


def validate(pred_classif, gt_classif, pred_nan=-1, gt_nan=-1, add_to_gtr=10000):
    """
    pred_classif and gt_classif *must* be aligned
    """
    gt_valid = np.logical_not(gt_classif == gt_nan)
    pred_valid = np.logical_not(pred_classif == pred_nan)
    valids = np.where(np.logical_and(gt_valid, pred_valid))

    # make sure the labels of gt_classif and pred_classif are different to get a meaningful confusion matrix
    # that's done here by adding add_to_gtr
    return valids, confusion_matrix(gt_classif.data[valids] + add_to_gtr, pred_classif.data[valids])
