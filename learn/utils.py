import xarray as xr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial


def run_jobs(jobs, joblib=True, n_jobs=4):
    if joblib:
        jobs = [delayed(job)() for job in jobs]
        out = Parallel(n_jobs=n_jobs, backend='threading')(jobs)
    else:
        out = []
        for job in jobs:
            out.append(job())
    return out


def pred_array(model, inp, arr=None, model_arr=None, batch_size=10000, no_val=-1, n_jobs=6):
    assert arr is not None or model_arr is not None
    if arr is None:
        arr = np.ones(model_arr.shape) * no_val
        shape = model_arr.shape
    if model_arr is None:
        shape = arr.shape

    # run in multiple batches for memory
    jobs = []
    inds = []
    for batch in np.array_split(inp, min(batch_size, len(inp))):
        jobs.append(partial(model.predict, batch.values))
        inds.append(batch.index)

    out = run_jobs(jobs, n_jobs=n_jobs)

    out_df = pd.DataFrame(index=inp.index)
    for i, pred in zip(inds, out):
        out_df.loc[i, :] =  pred
        arr[np.unravel_index(i, shape)] = pred

    arr = xr.DataArray(arr, dims=['y', 'x'])

    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr, out_df
