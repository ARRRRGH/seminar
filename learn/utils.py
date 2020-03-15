import xarray as xr
import numpy as np
import pandas as pd


def pred_array(model, inp, arr=None, model_arr=None, batch_size=10000, no_val=-1):
    assert arr is not None or model_arr is not None
    if arr is None:
        arr = np.ones(model_arr.shape) * no_val
    if model_arr is None:
        shape = arr.shape

    # run in multiple batches for memory
    out_df = pd.DataFrame(index=inp.index)
    
    for batch in np.array_split(inp, min(batch_size, len(inp))):
        tmp =  model.predict(batch.values)
        out_df.loc[batch.index, :] =  tmp
        arr[np.unravel_index(batch.index, model_arr.shape)] = tmp

    arr = xr.DataArray(arr, dims=['y', 'x'])

    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr, out_df
