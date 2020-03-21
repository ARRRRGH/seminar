import rasterio as rio
import numpy as np
from collections import OrderedDict
import os
import pandas as pd

from rasterio.windows import Window

try:
    from preproc.io import align
    import preproc.readers as rs
    import preproc.io as io
    from utils import run_jobs

except ModuleNotFoundError:
    from seminar.preproc.io import align
    import seminar.preproc.readers as rs
    import seminar.preproc.io as io
    from seminar.utils import run_jobs


def closest(lst, K):
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K))


def crop(rio_window, readers, label, path, mapping):
    label_path = os.path.join(path, str(label[i, j]))
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    patches = [reader.query(bbox=rio_window, xarray=False)[0][mapping, ...] for i, reader in enumerate(readers)]
    crop = np.stack(patches, axis=1)

    save_path = os.path.join(label_path, '_'.join(rio_window.bounds) + '.npy')
    np.save(save_path, crop)


def ind_to_bbox(ind, window_size, transform):
    i, j = ind
    top_left = transform * np.array([i - window_size // 2, j + window_size // 2])
    bottom_right = transform * np.array([i + window_size // 2, j - window_size // 2])

    minx, miny, maxx, maxy = top_left[0], bottom_right[1], top_left[1], bottom_right[1]

    return rs.BBox([minx, miny, maxx, maxy])


# Settings
out = '/home/PycharmProjects/seminar/notebooks/data/spv'
trn = os.path.join(out, 'trn')
val = os.path.join(out, 'val')

trn_size = 10000
val_size = 10000

window_size = 7

n_jobs = 8





# Read VV
# Read in data, crop first, then load from dir with cropped
reader_vv = rs.SeminarReader(time_pattern=r'_([0-9]+)_',
                             incl_pattern='.*',
                             dirpath='/home/jim/PycharmProjects/seminar/notebooks/data/vv_6d_12d_cropped')
arrs_vv, bboxs_vv = reader_vv.query(chunks=1000)
arrs_vv.name = 'vv'


# Read VH
# Read in data, crop first, then load from dir with cropped
reader_vh = rs.SeminarReader(time_pattern=r'_([0-9]+)_',
                             incl_pattern='.*',
                             dirpath='/home/jim/PycharmProjects/seminar/notebooks/data/vh_6d_12d_cropped')
arrs_vh, bboxs_vh = reader_vh.query(chunks=1000)
arrs_vh.name = 'vv'


# Read OPT
# Read in data, crop first, then load from dir with cropped
reader_opt = rs.SeminarReader(time_pattern=r'_([0-9]+)_',
                              incl_pattern='.*',
                              dirpath='/home/jim/PycharmProjects/seminar/notebooks/data/optical_model_cropped/')
arrs_opt, bboxs_opt = reader_opt.query(chunks=1000)
arrs_opt.name = 'opt'


# Read ground truth and align
ground_truth_classif, bboxs = io.read_raster('/home/jim/PycharmProjects/seminar/notebooks/' +
                                             'data/ground_truth_cropped/query_out/' +
                                             'pnrc_mos_2016_c_rasterized30x30m_UTM31N_cropped.tif')

ground_truth_classif.data = ground_truth_classif.astype(rio.int32)
ground_truth_classif = ground_truth_classif.expand_dims('band')
aligned_gt = align(arrs_vv, ground_truth_classif,
                   '/home/jim/PycharmProjects/seminar/notebooks/data/algined_ground_truth.tif')


# Assemble all readers
readers = [reader_vv, reader_vh, reader_opt]

# Bring ground truth in canonical form
uniques = np.unique(aligned_gt)
mapping = {code: val for val, code in enumerate(uniques)}
mapping[0] = -1
mapping[-1] = -1

mapper = np.vectorize(lambda entry: mapping.get(entry, entry))
aligned_gt.data = mapper(aligned_gt.data).astype(np.int8)


# Create map
map_sar_to_opt_time = OrderedDict([(i, closest(arrs_opt.coords['time'], t))
                                   for i, t in enumerate(arrs_vv.coords['time'])])

mapping = [(i, i, j) for i, j in map_sar_to_opt_time.items()] # since sar channels have equal times
mapping = zip(mapping)

# Create points
trn_points = pd.DataFrame({'y': np.random.randint(low=window_size // 2, high=arrs_vv.shape[2] - window_size // 2,
                                                  size=trn_size),
                           'x': np.random.randint(low=window_size // 2, high=arrs_vv.shape[3] - window_size // 2,
                                                  size=trn_size)})

val_points = pd.DataFrame({'y': np.random.randint(low=window_size // 2, high=arrs_vv.shape[2] - window_size // 2,
                                                  size=val_size),
                           'x': np.random.randint(low=window_size // 2, high=arrs_vv.shape[3] - window_size // 2,
                                                  size=val_size)})

# Create cropping jobs
jobs_trn = [crop(Window(ind[0], ind[1], window_size // 2, window_size // 2),
                 readers=readers, label=aligned_gt[ind[0], ind[1]], path=trn, map=mapping)
            for ind in trn_points.iterrows()]

jobs_val = [crop(ind_to_bbox(ind, window_size, arrs_vv.attrs['transform']),
                 readers=readers, label=aligned_gt[ind[0], ind[1]], path=val, map=mapping)
            for ind in val_points.iterrows()]

jobs = jobs_trn + jobs_val
run_jobs(jobs, n_jobs=n_jobs)
