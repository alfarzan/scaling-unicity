"""
This file contains the code to generate the learning curve plot (figure 2 on
the right in the main text)

It continueously samples larger and larger subsets of the real data, extracts
the input trajectories from that subset and then trains a model on that subset,
computing unicity values up to 1M generated trajectories.

Author: Ali Farzanehfar
"""


import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
from functools import partial
from collections import Counter
import unicity_utils as uut
import os
import multiprocessing as mp


def extract_time(u2p, lhrs, lants):
    """Extracts the time distribution from a given dataset u2p.

    Inputs:
        - u2p: dict, dictionary with keys being user ids and values being sets
               of points representing the trajectory
        - lhrs: int, total number of hours spanned by the data
        - lants: int, total number of locations spanned by the data

    Outputs:
        - time_arr: numpy array, represents the normalised circadian
                    distribution
    -------
    AF
    """
    time_arr = np.zeros(lhrs)
    get_track = partial(dut.get_user_track, lants=lants)
    for user in range(len(u2p)):
        tx = get_track(u2p[user])
        ti = tx[0]
        time_arr[ti] += 1
    time_arr = time_arr / time_arr.sum()
    return time_arr


def extract_activity(u2p, lhrs):
    """Extracts the activity distribution from a given dataset u2p.

    Inputs:
        - u2p: dict, dictionary with keys being user ids and values being sets
               of points representing the trajectory
        - lhrs: int, total number of hours spanned by the data

    Outputs:
        - actarr: numpy array, represents the normalised activity distribution
    -------
    AF
    """

    actarr = np.zeros(lhrs)
    for user in range(len(u2p)):
        a = len(u2p[user])
        actarr[a] += 1
    actarr = actarr / actarr.sum()
    return actarr


def extract_frequency(u2p, lhrs, lants, sgs=10):
    """Extracts the frequency distribution from a given dataset u2p.

    Inputs:
        - u2p: dict, dictionary with keys being user ids and values being sets
               of points representing the trajectory
        - lhrs: int, total number of hours spanned by the data
        - lants: int, total number of locations spanned by the data
        - sgs: int, size of the frequency array. Corresponds to the total
               number of distinct locations in each trajectory.

    Outputs:
        - mean_f: numpy array, represents the normalised frequency distribution
    -------
    AF
    """
    mean_f = np.zeros(lhrs)
    get_track = partial(dut.get_user_track, lants=lants)

    for user in range(len(u2p)):
        tx = get_track(u2p[user])
        xi = tx[1]
        md = Counter(xi)
        fi = sorted(md.items(), key=lambda tup: tup[1], reverse=True)
        fi = list(zip(*fi))[1]
        fi = np.array(fi)
        fi = fi / np.sum(fi)
        fi = np.pad(fi, (0, lhrs - len(fi)), 'constant')
        mean_f += fi
    mean_f = mean_f[:sgs] / mean_f[:sgs].sum()
    return mean_f


def get_inputs(u2p, lhrs, lants):
    """A wrapper function to extract the distributions from a given dataset:

    Inputs:
        - u2p: dict, dictionary with keys being user ids and values being sets
               of points representing the trajectory
        - lhrs: int, total number of hours spanned by the data
        - lants: int, total number of locations spanned by the data

    Outputs:
        - (a, f, t): tuple, returning 3 arrays corresponding to the
                     activity (a), frequency (f), and, circadian (t)
                     distributions
    -------
    AF
    """
    a = extract_activity(u2p, lhrs)
    f = extract_frequency(u2p, lhrs, lants)
    t = extract_time(u2p, lhrs, lants)
    return a, f, t


def get_all_inputs(u2parr, lhrs, lants, minsamp=1e2, maxsamp=1e4, nsamples=20):
    """
    A wrapper function that generates the list of distributions needed for the
    learning curve as well as the size of the population sample from which they
    were extracted.

    Inputs:
        - u2p: dict, dictionary with keys being user ids and values being sets
               of points representing the trajectory
        - lhrs: int, total number of hours spanned by the data
        - lants: int, total number of locations spanned by the data
        - minsamp: int, the size of the smallest population sample
        - maxsamp: int, the size of the largest population sample
        - nsamples: int, the total number of population samples

    Outputs:
        - inputs: list, containing the input distributions extracted from each
                  population sample
        - sampsizes: numpy array, containing the size of each population sample

    -------
    AF
    """
    allkeys = np.arange(0, len(u2parr))
    samplist = []
    sampsizes = np.linspace(minsamp, maxsamp, nsamples, dtype=int)
    for n in sampsizes:
        samp = np.random.choice(allkeys, replace=False, size=n)
        samplist.append(list(samp))

    inputs = []
    for arrs in tq(samplist):
        cu2p = u2parr[arrs]
        inputs.append(get_inputs(cu2p, lhrs, lants))
    return inputs, sampsizes


def gen_unicity_vals(inputs, max_size, step, sample_size=int(1e4),
                     seed=123456):
    """This function is a wrapper around the unicity computation function given a
    list of input distributions (inputs) as generated by get_all_inputs
    -------
    AF
    """
    dflist = []
    for inp in tq(inputs):
        df = uut.begin_unicity_series(
            max_size, step, sample_size, inp, seed=seed)
        dflist.append(df)
    return dflist


def worker(params):
    """Function for parallelising computation, can be ignored
    -------
    AF
    """
    max_size, step, sample_size, inp, seed, samppop, resd = params
    df = uut.begin_unicity_series(max_size, step, sample_size, inp, seed=seed)
    df.to_csv('{}iter_{:d}.csv'.format(resd, samppop))


def instantiate_pool(inputs, sampsizes, max_size, step, sample_size, seed,
                     nproc, resd):
    """Function for parallelising the computation, can be ignored
    -------
    AF
    """
    data = []
    for i in range(len(inputs)):
        di = (max_size, step, sample_size, inputs[i], seed, sampsizes[i], resd)
        data.append(di)

    mypool = mp.Pool(nproc)
    jobs = []
    for elem in data:
        jobs.append(mypool.apply_async(worker, args=(elem,)))

    mypool.close()

    for proc in tq(jobs):
        proc.get()

    mypool.join()
    pass


if __name__ == '__main__':
    minsamp = 1e2
    maxsamp = 5e4
    nsamples = 40
    nproc = 20
    max_size = int(1e6)
    sample_size = int(1e4)
    step = int(2e4)
    seed = 2018
    resd = '../results/learning_curve/'
    if not os.path.exists(resd):
        os.makedirs(resd)

    print('loading the data')
    allu2p = dut.get_u2p()
    lhrs = len(dut.get_date_array())
    lants = len(dut.get_ant_array())

    # remapping allu2p to list and then np arrays
    u2parr = list(np.zeros(len(allu2p)))
    for i in tq(allu2p):
        u2parr[i] = allu2p[i]
    u2parr = np.array(u2parr)

    print('getting the inputs')
    inputs, sampsizes = get_all_inputs(u2parr, minsamp, maxsamp, nsamples,
                                       lhrs, lants)
    np.save('{}sample_sizes.npy'.format(resd), sampsizes)
    print('begining processes')
    instantiate_pool(inputs, sampsizes, max_size,
                     step, sample_size, seed, nproc, resd)
