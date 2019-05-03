#!/usr/bin/env python
# coding: utf-8

# ## Load data

# In[ ]:


import os
import h5py
import json
import pandas as pd
from tqdm import tqdm
_
data_path = '/home/sgrill/repos/ml_heat/ml_heat/__data_store__/rawdata.hdf5'

with h5py.File(data_path, 'r') as file:
    organisation_ids = list(file['data'].keys())
    
    # take only animals that we have sensor data for
    eventlist = []
    for organisation_id in tqdm(organisation_ids):
        keys = list(file[f'data/{organisation_id}'].keys())
        animal_ids = [key for key in keys if key != 'organisation']
        for animal_id in animal_ids:
            a_keys = list(
                file[f'data/{organisation_id}/{animal_id}'].keys())
            if len(a_keys) < 2:
                continue
            animal = json.loads(file[f'data/{organisation_id}/{animal_id}/animal'][()])
            cycles = animal.get('lifecycle', {}).get('cycles', [])
            for cycle in cycles:
                events = cycle.get('events', [])
                for event in events:
                    eventlist.append(event)

frame = pd.DataFrame(eventlist)
frame


# In[ ]:


frame[frame.event_type.isin(['insemination', 'heat'])].sort_values(by='event_ts')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

series = pd.to_datetime(frame.event_ts)

plt.hist(series, 50)

