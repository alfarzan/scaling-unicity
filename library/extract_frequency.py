"""
This file extracts the frequency vectory for each user in the data and puts
the values in a histogram

Author: Ali Farzanehfar
"""

import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
from functools import partial
from collections import Counter


u2p = dut.get_u2p()

lhrs = len(dut.get_date_array())
lants = len(dut.get_ant_array())

mean_f = np.zeros(lhrs)

get_track = partial(dut.get_user_track, lants=lants)

for user in tq(u2p):

    _, xi = get_track(u2p[user])
    md = Counter(xi)
    fi = sorted(md.items(), key=lambda tup: tup[1], reverse=True)
    fi = list(zip(*fi))[1]
    fi = np.array(fi)
    fi = fi / np.sum(fi)
    fi = np.pad(fi, (0, lhrs - len(fi)), 'constant')
    mean_f += fi


np.save('../inputs/frequency.npy', mean_f)
