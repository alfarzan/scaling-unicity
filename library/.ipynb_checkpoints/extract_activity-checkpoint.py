"""
This file extracts the activity for each user in the data and puts the values
in a list to be fit later.

Author: Ali Farzanehfar
"""
import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
import pickle


u2p = dut.get_u2p()

activity = []
actarr = np.zeros(len(dut.get_date_array()))
for user in tq(u2p):
    a = len(u2p[user])
    activity.append(a)
    actarr[a] += 1

with open('../inputs/activity.p', 'wb') as actfile:
    pickle.dump(activity, actfile)

# this is the one of interest and the one used in the model
np.save('../inputs/activity.npy', actarr)
