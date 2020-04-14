import xarray as xr
import numpy as np
import pandas as pd
from functools import partial

try:
    from utils import run_jobs
except ModuleNotFoundError:
    import seminar
    from seminar.utils import run_jobs


def pred_array(model, inp, arr=None, model_arr=None, batch_size=10000, no_val=-1, n_jobs=6):
    assert arr is not None or model_arr is not None
    if arr is None:
        arr = np.ones(model_arr.shape) * no_val
        shape = model_arr.shape
    if model_arr is None:
        shape = arr.shape

    inp_is_list = False
    if inp is list:
        orig_inp = inp
        inp = inp[0]
        inp_is_list = True

    # run in multiple batches for memory
    jobs = []
    inds = []
    for batch in np.array_split(inp, min(batch_size, len(inp))):
        index = batch.index

        if inp_is_list:
            batch = [ip.loc[index].values for ip in orig_inp]
        else:
            batch = batch.values

        jobs.append(partial(model.predict, batch))
        inds.append(index)

    out = run_jobs(jobs, n_jobs=n_jobs)

    out_df = pd.DataFrame(columns=['pred'], index=inp.index)
    out_df = out_df.fillna(-1)

    for i, pred in zip(inds, out):
        out_df.loc[i, 'pred'] = pred
        arr[np.unravel_index(i, shape)] = pred
    arr = xr.DataArray(arr, dims=model_arr.dims)

    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr, out_df
