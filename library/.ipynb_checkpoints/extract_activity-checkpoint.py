"""
This file extracts the activity for each user in the data and puts the values
in a list to be fit later.
"""
import dataformat_utils as dut
from tqdm import tqdm as tq
import pickle

u2p = dut.get_u2p()

activity = []
for user in tq(u2p):
    activity.append(len(u2p[user]))


with open('../inputs/activity.p', 'wb') as actfile:
    pickle.dump(activity, actfile)
