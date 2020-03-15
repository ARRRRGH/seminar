#!/usr/bin/env python
# coding: utf-8

# In[1]:


SEMINAR_PATH = '/home/jim/PycharmProjects'
import sys
sys.path.append(SEMINAR_PATH)


import seminar.preproc.readers as rs



from seminar.preproc.base_data_structures import BBox
from fiona.crs import from_epsg

from datetime import datetime




# ## Load Data

# In[3]:

bbox = BBox((5.905e5, 4.835e6, 6.6997e5, 4.798e5), crs=from_epsg(32631))


# In[5]:


reader2 = rs.SeminarReader(time_pattern=r'_([0-9]+)_',
                          incl_pattern='.*_model.*',
                          dirpath='/home/jim/mount/shared/course/geo441/data/2020_Camargue/optical')


# In[6]:


arrs_opt_models, bboxs_opt_model = reader2.query(bbox=bbox, time=(datetime(year=2017, day=1, month=1), None),
                                                 out=True, out_dir='/home/jim/PycharmProjects/seminar/notebooks/data/test_out')


