import numpy as np
import pandas as pd


def binarize_dataframe(dframe, var, vals, pad_lo=None, pad_hi=None):
    if pad_lo is not None:
        vals = np.concatenate((np.array([pad_lo]), vals))
    if pad_hi is not None:
        vals = np.concatenate((vals, np.array([pad_hi])))

    bin_edges = vals[:-1] + (vals[1:] - vals[:-1]) / 2
    idxs = pd.cut(dframe[var], bins=bin_edges, labels=False)

    _binned_dframes = {}
    for i in np.unique(idxs):
        dfi = dframe.iloc[np.where(idxs == i)[0]]
        _binned_dframes[i] = dfi

    return _binned_dframes

