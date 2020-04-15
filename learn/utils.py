import xarray as xr
import numpy as np
import pandas as pd
from functools import partial

try:
    from utils import run_jobs
    from preproc.transformers import InputList
except ModuleNotFoundError:
    import seminar
    from seminar.utils import run_jobs
    from seminar.preproc.transformers import InputList


def pred_array(model, inp, arr=None, model_arr=None, n_batches=1000, no_val=-1, n_jobs=6):
    assert arr is not None or model_arr is not None
    if arr is None:
        arr = np.ones(model_arr.shape) * no_val
        shape = model_arr.shape
    if model_arr is None:
        shape = arr.shape

    inp_is_list = False
    if type(inp) is InputList:
        inp_is_list = True

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
            return (spl_model.index, InputList([inp.get(k).loc[spl_model.index].values
                                                for k in range(len(inp))]))

        orig_index = inp.get(0).index
        n_batches = len(split_model)

    def predict_batch(nth_batch):
        index, batch = split(nth_batch)
        return index, model.predict(batch)

    jobs = []
    for j in range(n_batches):
        jobs.append(partial(predict_batch, nth_batch=j))

    res = run_jobs(jobs, n_jobs=n_jobs)

    out_df = pd.DataFrame(columns=['pred'], index=orig_index)
    out_df = out_df.fillna(no_val)

    for i, pred in res:
        out_df.loc[i, 'pred'] = pred
        arr[np.unravel_index(i, shape)] = pred

    arr = xr.DataArray(arr, dims=model_arr.dims)
    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr, out_df
