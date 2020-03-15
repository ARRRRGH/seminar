import xarray as xr
import numpy as np

def pred_array(model, inp, arr=None, model_arr=None, batch_size=10000, no_val=-1):
    assert arr is not None or model_arr is not None
    if arr is None:
        arr = np.ones(model_arr.shape) * no_val
    if model_arr is None:
        shape = arr.shape

    # run in multiple batches for memory
    for batch in np.array_split(inp, min(batch_size, len(inp))):
        arr[np.unravel_index(batch.index, model_arr.shape)] = model.predict(batch.values)

    arr = xr.DataArray(arr, dims=['y', 'x'])

    arr.attrs = model_arr.attrs.copy()
    arr = arr.assign_coords(model_arr.coords)

    return arr
