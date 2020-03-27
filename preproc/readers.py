import os
import glob
import datetime as dt
import re

from RSreader.base import _TimeRasterReader


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
