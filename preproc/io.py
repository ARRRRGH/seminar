try:
    from preproc.readers import _RasterReader
except ModuleNotFoundError:
    import seminar
    from seminar.preproc.readers import _RasterReader

import rasterio as rio


def read_raster(path, bbox=None, *args, **kwargs):
    return _RasterReader(path, bbox).query(*args, **kwargs)


def write_out(arr, dst_path, default_meta):
    meta = default_meta.copy()
    meta.update({
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': arr.dtype,
        'crs': meta['crs'],
    })
    del meta['path']

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rio.Env():
        with rio.open(dst_path, 'w', **meta) as dst:
            dst.write(arr.data.astype(rio.int8), 1)

    arr.attrs['path'] = dst_path
    return arr


def align(arr1, arr2, path_wrp, no_data=-1):
    """
    Align arr2 to arr1
    """
    # align classificication to ground truth
    kwargs = arr2.attrs.copy()
    kwargs.update({
        'crs': arr1.attrs['crs'],
        'transform': arr1.attrs['transform'],
        'width': arr1.shape[1],
        'height': arr1.shape[0],
        'driver': 'GTiff',
        'count': 1,
        'dtype': arr2.dtype,
    })
    try:
        del kwargs['path']
    except KeyError:
        pass

    with rio.open(path_wrp, 'w', **kwargs) as dst:
        rio.warp.reproject(
            source=arr2.data,
            destination=rio.band(dst, 1),
            src_transform=arr2.transform,
            src_crs=arr2.crs,
            dst_transform=arr1.transform,
            dst_crs=arr1.crs,
            resampling=rio.warp.Resampling.nearest,
            dst_nodata=no_data,
            dst_dtype=arr2.dtype)

    tmp_reader = _RasterReader('')
    arrs, bboxs = tmp_reader.query(paths=path_wrp)

    algned = arrs[0]
    algned = algned.assign_coords(arr1.coords)
    algned.attrs = arr1.attrs
    algned.attrs['path'] = path_wrp

    return algned
