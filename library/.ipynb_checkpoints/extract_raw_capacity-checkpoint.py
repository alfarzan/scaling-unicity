"""
This file extracts the activity, capacity and weekly time coverage for each user
"""
import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
from numba import jit


@jit(nopython=True)
def count_unique(a):
    return len(set(a))

@jit(nopython=True)
def count_weeks(t, nwks):
    t = t // 168
    cnt = len(set(t))
    return cnt / nwks


u2p = dut.get_u2p()

act = np.zeros(len(u2p), dtype=np.int32)  # activity
cap = np.zeros(len(u2p), dtype=np.int32)  # capacity
tc = np.zeros(len(u2p))  # time coverage (in wks)

nants = len(dut.get_ant_array())
lhrs = len(dut.get_date_array())
nwks = lhrs // (24 * 7) + 1  # total number of wks in data

i = 0
for user in tq(u2p):
    trace = u2p[user]
    t, x = dut.get_user_track(trace, nants)
    act[i] = len(x)
    cap[i] = count_unique(x)
    tc[i] = count_weeks(t, nwks)
    i += 1

np.save('../inputs/raw_activity.npy', act)
np.save('../inputs/raw_capacity.npy', cap)
np.save('../inputs/weekly_time_coverage.npy', tc)