"""
This file extracts the circadian distribution from the data with differential privacy (DP).

For reference, the non-DP method is in library/extract_time.py.

Method:
 - The data for each player is as a list of positions and times (a "track").
 - We bound the sensitivity of each part by selecting K random entries of the trajectories.
 - Let (x_i, t_i) for 1 <= i <= k' <= K the samples for a user (k' < K if the user has less samples).
 - We then compute a histogram of the time of user contributions, defined as:
    time_arr[hour] = sum_u sum_i I{t_i = hour}
 - We then add Laplace noise to each bucket, with scale sensitivity/epsilon.
    - where sensitivity = K (since there are only K element per bucket).
 - We finally post-process the noisy histogram, setting negative values to 0, rounding to integers,
     then normalizing so the vector sums to 1 (and is thus a probability distribution over hours).

Authors: Ali Farzanehfar and Florimond Houssiau
"""

import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
from functools import partial


# PRIVACY PARAMETERS.
# An epsilon of 1 is a low value, usually considered secure.
PRIV_EPSILON = 1
# Low bounds create bias, high bounds require lots of noise.
# This seems like a reasonable compromise.
PRIV_CONTRIBUTION_BOUND = 50



# Get samples from the dataset.
u2p = dut.get_u2p()

# Number of antennas and hours (non sensitive).
lhrs = len(dut.get_date_array())
lants = len(dut.get_ant_array())

# User contribution histogram (the result).
time_arr = np.zeros(lhrs)

get_track = partial(dut.get_user_track, lants=lants)

# Build the histogram for each user.
for user in tq(u2p):

    times, _ = get_track(u2p[user])
    n_times = len(times)
    # Limit any user's contributions to the result
    if n_times > PRIV_CONTRIBUTION_BOUND:
        indices = np.arange(n_times)
        np.random.shuffle(indices)
        chosen_samples = indices[:PRIV_CONTRIBUTION_BOUND]
        # Restrict the `times` vector to the chosen samples.
        times = times[chosen_samples]
    # Increase by 1 the count for each element in the remaining array.
    time_arr[times] += 1


# Add Laplace noise to the histogram.
dp_noise = np.random.laplace(loc=0, scale=PRIV_CONTRIBUTION_BOUND/PRIV_EPSILON, size=(lhrs,))

dp_time_arr = time_arr + dp_noise


# Post-processing: round, and normalize.
# This is needed to ensure that the results are usable.
#  1. Set negative entries to 0 (project on R^n+).
dp_time_arr[dp_time_arr < 0] = 0
#  2. Round to integers (this is a neat trick to prevent the LSB attack from Mironov et al.
#     "On significance of the least significant bits for differential privacy")
dp_time_arr = np.round(dp_time_arr)
#  3. Normalise to a probability distribution.
dp_time_arr = dp_time_arr / dp_time_arr.sum()


np.save('../inputs/circadian.npy', dp_time_arr)