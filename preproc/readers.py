#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:32:26 2019

@author: jim
"""
import pandas as pd
import geopandas as gpd
import os
# import h5py
import rasterio as rio
from shapely.geometry import Point, GeometryCollection
from astropy.time import Time
import glob
import numpy as np
import datetime as dt
import re
from rasterio.mask import mask
from joblib import Parallel, delayed
from . import base_data_structures as bs
from . import helpers as hp

import xarray as xr
import uuid
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from fiona.crs import from_epsg

#from tqdm import tqdm
tqdm = lambda x: x

from shapely import wkt
import pickle as pkl


def rasterio_to_xarray(arr, meta, tmp_dir='.', fil_name=None, chunks=None, out=False, *args, **kwargs):
    if fil_name is None:
        fil_name = str(uuid.uuid4())
    tmp_path = os.path.join(tmp_dir, '%s' % fil_name)

    with rio.open(tmp_path, 'w', **meta) as fil2:
        fil2.write(arr)
    ret = xr.open_rasterio(tmp_path, chunks=chunks)
    ret = clean_raster_xarray(ret)

    if chunks is None and not out:
        os.remove(tmp_path)

    return ret, tmp_path


def clean_raster_xarray(ret):
    if len(ret.band) == 1:
        ret = ret.squeeze('band', drop=True)
    ret = ret.where(ret != ret.attrs['nodatavals'][0])
    ret.attrs['nodatavals'] = np.nan
    return ret


class _Reader(object):
    def __init__(self, path, bbox=None, time=None, *args, **kwargs):
        self.path = path
        self.bbox = bbox
        self.time = time

    def _which_bbox(self, bbox):
        if bbox is None:
            bbox = self.bbox
        return bbox

    def _which_time(self, time):
        if time is None:
            time = self.time
        return time

    def query(self, time=None, bbox=None, n_jobs=2, crs=None, *args, **kwargs):
        pass


class _RasterReader(_Reader):
    """
    _RasterReader is an interface between SWISSMap and rasterio reading. It handles queries related to raster operations
    during reading. Readers specific to some data type or directory structure build on top of _RasterReader.
    """

    def __init__(self, path, bbox=None, time=None, *args, **kwargs):
        _Reader.__init__(self, path, bbox=bbox, time=time)

    def read(self, paths=None, bbox=None, align=False, crs=None, chunks=None, fil_names=None,
             out=False, out_dir='~', *args, **kwargs):
        """
        :param paths:
        :param bbox:
        :param args:
        :param kwargs:
        :return:
        """
        bbox = self._which_bbox(bbox)

        # single file read
        if paths is None:
            paths = self.path
        if type(paths) is str:
            paths = [paths]

        out_xarrs = []
        out_bboxs = []

        if align or bbox is not None:
            if bbox is None:
                bbox = bs.BBox.from_tif(paths[0])
            for i, path in tqdm(enumerate(paths)):
                if fil_names is None:
                    basename = os.path.dirname(path)
                    new_dir = os.path.join(out_dir, 'query_out')
                    os.makedirs(new_dir, exist_ok=True)
                    fil_name, ext = os.path.splitext(os.path.basename(path))
                    #print(new_dir, fil_name, ext)
                    fil_name = os.path.join(new_dir, fil_name + '_cropped' + ext)
                else:
                    fil_name = fil_names[i]

                # crop tif and save to tmp file
                ret, tmp_path, fil_crs = _RasterReader._crop_tif(path, bbox=bbox, chunks=chunks, out=True, fil_name=fil_name, *args, **kwargs)

                crs = crs if crs is not None else fil_crs

                # warp image
                if crs != fil_crs:
                    fil, ext = os.path.splitext(os.path.basename(fil_name))
                    fil_name = os.path.join(os.path.dirname(fil_name), fil + '_warped' + ext)
                    ret, tmp_path2 = _RasterReader._warp_tif(tmp_path, bbox=bbox, crs=crs, chunks=chunks,
                                                             fil_name=fil_name, out=out, *args, **kwargs)

                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    tmp_path = tmp_path2

                ret.attrs['crs'] = dict(rio.crs.CRS.from_string(ret.crs))
                ret.attrs['path'] = tmp_path
                
                if not out:
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

                if not out:
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

                out_xarrs.append(ret)
                out_bboxs.append(bbox)

        else:
            for path in tqdm(paths):
                # fil_name = os.path.basename(path)
                # if bbox is not None:
                #     ret, path = _RasterReader._crop_tif(path, bbox=bbox, chunks=chunks, fil_name=fil_name,
                #                                      out=out, *args, **kwargs)
                #     out_bbox = bbox
                # else:
                with rio.open(path, 'r') as fil:
                    ret = xr.open_rasterio(fil, chunks=chunks)
                    ret = clean_raster_xarray(ret)

                # make bbox that is returned
                out_bbox = bs.BBox.from_tif(path)

                ret.attrs['crs'] = dict(rio.crs.CRS.from_string(ret.crs))
                ret.attrs['path'] = path

                out_xarrs.append(ret)
                out_bboxs.append(out_bbox)

        return out_xarrs, out_bboxs

    @staticmethod
    def _crop_tif(path, bbox, tmp_dir='.', fil_name=None, *args, **kwargs):
        with rio.open(path) as fil:
            coords = bbox.get_rasterio_coords(fil.crs.data)
            out_img, out_transform = mask(dataset=fil, shapes=coords, crop=True)
            out_meta = fil.meta.copy()
            crs = fil.crs

            out_meta.update({"driver": "GTiff",
                             "height": out_img.shape[1],
                             "width": out_img.shape[2],
                             "transform": out_transform,
                             "count": fil.count,
                             "dtype": out_img.dtype})

        out, tmp_path = rasterio_to_xarray(out_img, out_meta, tmp_dir=tmp_dir,
                                           fil_name=fil_name, *args, **kwargs)

        return out, tmp_path, crs

    @staticmethod
    def _warp_tif(path, bbox, crs, tmp_dir='.', fil_name=None, resampling_method='cubic', *args, **kwargs):
        left, bottom, right, top = bbox.get_bounds(crs=crs)
        res = bbox.get_resolution(crs)

        width = (right - left) // res[0]
        height = (top - bottom) // res[1]
        dst_transform = rio.transform.from_origin(west=left, north=top, xsize=res[0], ysize=res[1])

        vrt_options = {
            'resampling': Resampling[resampling_method],
            'crs': crs,
            'transform': dst_transform,
            'height': height,
            'width': width,
        }

        with rio.open(path) as src:
            with WarpedVRT(src, **vrt_options) as vrt:
                # At this point 'vrt' is a full dataset with dimensions,
                # CRS, and spatial extent matching 'vrt_options'.
                dta = vrt.read()
                # # Read all data into memory.
                # #
                vrt_meta = vrt.meta.copy()

                vrt_meta.update({"driver": "GTiff",
                                 "height": dta.shape[1],
                                 "width": dta.shape[2],
                                 "transform": dst_transform,
                                 "count": vrt.count})

                xarr, tmp_path = rasterio_to_xarray(dta, vrt_meta, tmp_dir=tmp_dir,
                                                    fil_name=fil_name, *args, **kwargs)

        return xarr, tmp_path

    def query(self, time=None, bbox=None, n_jobs=2, crs=None, align=False, *args, **kwargs):
        ret, bbox = self.read(time=time, bbox=bbox, n_jobs=n_jobs, crs=None, align=align, *args, **kwargs)

        # if crs is set, change coordinates
        # fixme: incorrect transformation ?
        # if crs is not None and not align:
        #     ret = [hp.xarray_to_crs(r, crs) for r in ret]

        return ret, bbox


class _TimeRasterReader(_RasterReader):
    """
    _TimeRasterReader handles data sets of rasters with a time label. Time information extraction takes place in
    _create_path_dict.
    """

    def __init__(self, dirpath, bbox=None, time=None, *args, **kwargs):
        _RasterReader.__init__(self, path=dirpath, bbox=bbox, time=time, *args, **kwargs)

        self._path_dict = self._create_path_dict()
        self.min_time = min(self._path_dict.values())
        self.max_time = max(self._path_dict.values())

    def query(self, bbox=None, time=None, align=False, crs=None, *args, **kwargs):
        paths, times = self._prepare_query(time=time)
        arrs, bboxs = self.read(paths, bbox=bbox, align=align, crs=crs, *args, **kwargs)

        ret = xr.concat(arrs, 'time')
        ret.coords['time'] = ('time', np.array(times))
        ret = ret.sortby('time')

        str_times = np.array(times, dtype=np.datetime64).astype(str).astype('<U13')
        ret.attrs['path'] = dict([(t, a.attrs['path']) for t, a in zip(str_times, arrs)])

        return ret, bboxs

    def _prepare_query(self, time=None, *args, **kwargs):
        time = self._which_time(time)

        if time is None:
            pathes_times = list(self._path_dict.items())
        else:
            start, end = time
            if start is None:
                start = self.min_time
            if end is None:
                end = self.max_time
            pathes_times = list((path, time) for path, time in self._path_dict.items() if start <= time <= end)

        if len(pathes_times) == 0:
            return None, None

        paths, times = zip(*pathes_times)
        return paths, times

    def _create_path_dict(self):
        pass


class SeminarReader(_TimeRasterReader):
    def __init__(self, time_pattern, incl_pattern='.*', *args, **kwargs):
        self.time_pattern = time_pattern
        self.incl_pattern = incl_pattern
        _TimeRasterReader.__init__(self, *args, **kwargs)

    def _create_path_dict(self):
        fnames = glob.glob(os.path.join(self.path, '*.tif'))
        
        acc = []
        for f in fnames:
            if len(re.findall(self.incl_pattern, f)) != 0:
                acc.append(f)

        path_dict = {}
        for i, fname in enumerate(acc):
            date = re.findall(self.time_pattern, fname)[-1]
            year = int(date[:4])
            month = int(date[4:6])
            day = int(date[6:8])
            path_dict[os.path.join(self.path, fname)] = dt.datetime(year=year, month=month, day=day)

        return path_dict



def read_raster(path, bbox=None, *args, **kwargs):
    return _RasterReader(path, bbox).query(*args, **kwargs)
