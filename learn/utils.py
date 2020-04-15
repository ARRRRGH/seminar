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


def pred_array(model, inp, arr=None, model_arr=None, batches=1000, no_val=-1, n_jobs=6):
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
    jobs = []
    inds = []

    if not inp_is_list:
        splits = [(spl.index, spl.values) for spl in np.array_split(inp, min(batches, len(inp)))]
        split = lambda : splits
        orig_index = inp.index
    else:
        split_model = np.array_split(inp.get(0), min(batches, len(inp.get(0))))

        def split():
            for spl_model in split_model:
                yield (spl_model.index, InputList([inp.get(j).loc[spl_model.index].values
                                                             for j in range(len(inp))]))
        orig_index = inp.get(0).index

    for index, batch in split():
        jobs.append(partial(model.predict, batch))
        inds.append(index)

    out = run_jobs(jobs, n_jobs=n_jobs)

    out_df = pd.DataFrame(columns=['pred'], index=orig_index)
    out_df = out_df.fillna(no_val)

    for i, pred in zip(inds, out):
        out_df.loc[i, 'pred'] = pred
        arr[np.unravel_index(i, shape)] = pred
    arr = xr.DataArray(arr, dims=model_arr.dims)

    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr, out_df
