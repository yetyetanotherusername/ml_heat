#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import vaex as vx

vxstore = '/home/sgrill/Documents/repos/master_project/ml_heat/ml_heat/__data_store__/vaex_store'

def load_vxframe():
    vxfiles = [
        os.path.join(vxstore, x) for x in
        os.listdir(vxstore) if x.endswith('.hdf5')]
    frame = vx.open(vxfiles[0])
    for file in vxfiles[1:]:
        frame2 = vx.open(file)
        frame = frame.merge()
    return frame

frame = load_vxframe()
frame


# In[36]:


frame = vx.open('/home/sgrill/Documents/repos/master_project/ml_heat/ml_heat/__data_store__/vaex_store/one_hot.hdf5')
frame.describe()


# In[39]:


import numpy as np
x = np.arange(5)
y = x ** 2
z = y ** 2
frame = vx.from_arrays(x=x, y=y, z=z)

frame['x', 'y'].export('test.hdf5')
frame['y', 'z'].export('test2.hdf5')

frame1 = vx.open('test.hdf5')
frame2 = vx.open('test2.hdf5')

frame = frame1.merge(frame2)


# In[43]:


frame = vx.open('/home/sgrill/Documents/repos/master_project/ml_heat/ml_heat/__data_store__/vaex_store/one_hot.hdf5')
frame

