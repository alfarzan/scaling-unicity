"""
This file contains all the functions that are used to compute unicity.
This includes everything from small functions that check whether a user is
unique based on a set of points, all the way to functions that compute
the unicity on an entire data set using sparse matrix representations.

Author: Ali Farzanehfar (AF)
"""

import numpy as np
import os
from scipy import sparse as sps
from dataformat_utils import sparsify_mat_list, vstack_multiply
from dataformat_utils import chunkify_mat_list
from geoloc_utils import get_geo
from model_source import create_cluster_array, resampler
from collections import defaultdict
import pandas as pd
import random as rnd
from tqdm import tqdm as tq


def get_sample(u2p, sample_size, seed=None):
    """Randomly samples a part of the original u2p matrix.

    Inputs:
        - u2p: 5-tuple, output of resampler
        - sample_size: int, specifies the size of population to be sampled
        - seed: int, a seed for the numpy random number generator

    Outputs:
        - sample: 5-tuple similar to output of resampler
    -------
    AF
    """

    if seed is not None:
        np.random.seed(seed)

    data, rows, cols, shape, rand_acts = u2p
    n, p = shape
    pop = np.random.choice(np.arange(n, dtype=np.int32),
                           size=sample_size, replace=False)
    sacts = rand_acts[pop]

    start_inds = np.cumsum(rand_acts, dtype=np.int32)
    ss = (sample_size, p)  # sample shape

    ll = sacts.sum()
    sr = np.ones(ll, dtype=np.int32)  # rows
    sc = np.zeros(ll, dtype=np.int32)  # cols
    sd = np.ones(ll, dtype=np.int8)  # data

    cind = 0
    for user in range(sample_size):
        ind = pop[user]
        a = sacts[user]
        sr[cind:cind + a] = user * sr[cind:cind + a]
        start = start_inds[ind]
        sc[cind:cind + a] = cols[start - a:start]
        cind += a
    return sd, sr, sc, ss, sacts


def begin_unicity_series(max_size, step, sample_size, inputs,
                         pl=[2, 3, 4, 5], cs=int(1e4), sgs=10, seed=None,
                         autosave=False, verbose=False):
    """computes the unicity from 'sample_size' to 'max_size' population in
    steps.

    Inputs:
        - max_size: int, maximum population size to estimate unicity for
        - step: int indication the size of users to be generated (and then
          deleted) at each step of the calculation
        - sample_size: int, indicates the sample size
        - pl: list of ints indicating the number of points to be included for
          unicity calculation
        - inputs: 3-tuple of numpy arrays similar to the output of
          get_input_dists()
        - cs: int, chunk_size which represents the size of chunks for sparse
          matrices to be generated
        - sgs: int, the size of antenna clusters used for each user
        - seed: int, seed for both numpy.random.seed and random.seed
        - autosave: bool or str, if not False then after each loop a copy of the dataframe
          is saved to the temporary folder autosave.
        - verbose: bool, if true then display some information about the current
          status of the computation.

    Outputs:
        - df: pandas.DataFrame() object which contains the results of the
          calculation. Each row represnts a population size while each column
          represnts the unicity for a particular number of points.
          It is indexed by the population values.
    -------
    AF
    """
    if seed is not None:
        np.random.seed(seed)
        rnd.seed(seed)

    # Create the target folder, if needed.
    if autosave:
        if not os.path.exists(autosave):
            os.mkdir(autosave)

    fprint = print if verbose else lambda *x, **y: None  # Logging function

    # getting geographical inputs
    fprint('Loading geographical inputs...')
    ana = get_geo('../inputs/', 'location_grid.txt')


    # generating the first step
    fprint('Generating clusters...')
    carr = create_cluster_array(step, sgs, ana)
    s_u2p = resampler(step, carr, inputs, ana)

    pop_list = np.arange(step, max_size + step, step, dtype=np.int32)
    nsteps = len(pop_list)

    # generating the samples
    sample_seeds = np.random.permutation(nsteps)

    # creating the results dataframe
    vals = np.zeros((len(pl), nsteps))
    inputdict = dict(zip(pl, vals))
    df = pd.DataFrame(inputdict)
    df = df.set_index(pop_list)

    # creating the housing for the colsums
    colsum_dict = {}
    for point in pl:
        colsum_dict[point] = np.zeros((nsteps, sample_size))

    for iii in range(nsteps):

        fprint('\rStep %d/%d...' % (iii+1,nsteps), end='')

        # if it's the first one make sure to not regenerate
        if iii != 0:
            carr = create_cluster_array(step, sgs, ana)
            u2p = resampler(step, carr, inputs, ana)
        else:
            u2p = s_u2p

        ml = chunkify_mat_list(u2p, cs)
        sml = sparsify_mat_list(ml)

        for jjj in range(iii, nsteps):
            sample = get_sample(s_u2p, sample_size, sample_seeds[jjj])
            smats = get_random_points(pl, sample, sample_seeds[jjj])
            ps = vstack_multiply(sml, smats)

            # computing the unicity
            for point in pl:
                csum = ps[point] / point
                csum = csum.floor()
                csum = csum.sum(axis=0)
                colsum_dict[point][jjj] += np.array(csum)[0]

        for point in pl:
            u = np.count_nonzero(colsum_dict[point][iii] == 1) / sample_size
            df[point].loc[pop_list[iii]] = u
        if autosave:
            df.to_csv(os.path.join(autosave, 'tmp.csv'))

    fprint('\nDone!')
    return df


def get_random_points(pl, sample, seed=None):
    """Samples a fix number of rows from a population. From each row, it samples
    a fixed number of points.

    Inputs:
        - pl: list of ints, specifies the number of points to be sampled from
          each row
        - sample: 5-tuple returned by resampler
        - seed: int, seed for the numpy random number generator

    Outputs:
        - smats: a dict of scipy.sparse.csr_matrix() objects containing the
          sample for each number of points. Dict keys represent the number of
          points
    -------
    AF
    """
    if seed is not None:
        np.random.seed(seed)
    data, rows, cols, shape, rand_acts = sample
    n, p = shape
    smat_list = defaultdict(list)
    for cp in pl:
        smat_list[cp].append(np.ones(n * cp, dtype=np.int8))  # data
        smat_list[cp].append(np.zeros(n * cp, dtype=np.int32))  # row
        smat_list[cp].append(np.ones(n * cp, dtype=np.int32))  # cols
        smat_list[cp].append((p, n))  # shape
    smat_list = dict(smat_list)

    col_starts = np.cumsum(rand_acts, dtype=np.int32)
    currind = defaultdict(int)
    for uid in range(n):
        a = rand_acts[uid]
        start = col_starts[uid]
        for cp in pl:
            currows = np.random.choice(cols[start - a:start], size=cp,
                                       replace=False)
            currcols = uid * smat_list[cp][2][currind[cp]:currind[cp] + cp]
            smat_list[cp][1][currind[cp]:currind[cp] + cp] = currows
            smat_list[cp][2][currind[cp]:currind[cp] + cp] = currcols
            currind[cp] += cp

    smats = {}
    for cp in pl:
        d, r, c, s = smat_list[cp]
        smat = sps.csr_matrix((d, (r, c)), shape=s)
        smats[cp] = smat
    return smats


def check_unique(pop, pset, p2u):
    """Checks whether a set of points are unique in a population.

        Inputs:
            - pop: set, represnts the whole population under consideration
            - pset: set, a set of points corresponding the auxilary information
                    in unicity computations
            - p2u: dict, keys being points and values being sets of user ids.

        Output:
            - a binary integer with 1 coresponding to trajectory being unique
    -------
    AF
    """
    for p in pset:
        pop = pop.intersection(p2u[p])
    return int(len(pop) == 1)


def get_sample_and_pop(alluserids, smin, smax, step, sample_size=int(1e4)):
    """This is a helper function that generates 1) a list of sets of increasing
    size containing population user IDs, 2) a list of sets each containing 10K
    user IDs which are the samples and, 3) a numpy array containing the sizes
    of the sets in each population set in 1.

    Inputs:
        - alluserids: list, contains all user ids in the entire dataset
        - smin: int, the size of the smallest population sample to be chosen
        - smax: int, the size of the largest population sample to be chosen
        - step: int, the step size from smin to smax for generating populations
        - sample_size, int, the size of the sample for computing unicity

    Outputs:
        - pop_ids: list, a list of sets of increasing size each containing user
                   ids
        - sample_ids: list, a list of sets of size 10K each containing user ids
                      sampled from the corresponding set in pop_ids
        - sizes: numpy array, an array of integers containing the sizes of sets
                 in pop_ids
    -------
    AF
    """
    sizes = np.arange(smin, smax, step, dtype=int)  # inclusive edges
    pop_ids = []
    sample_ids = []
    for s in tq(sizes):
        pop = np.random.choice(alluserids, size=s, replace=False)
        samp = np.random.choice(pop, size=sample_size, replace=False)
        pop_ids.append(set(pop))
        sample_ids.append(set(samp))
    return pop_ids, sample_ids, sizes


def compute_unicity(u2p, p2u, popids, sampids, point_list=[2, 3, 4, 5]):
    """This computes the unicity for a given population size for varying
    numbers of points of side information (defualt being [2, 3, 4, 5]).

    Inputs:
        - u2p: dict, dictionary with keys being user ids and values being sets
               of points representing the trajectory
        - p2u: dict, dictionary with keys being points and values being sets
               of user ids which have visited a given point
        - popids: list, list containing all users in a given population
        - sampids: list, a sample of size 10K from popids
        - point_list: list, a list of points representing the number of points
                      used as side information in the unicity computation

    Outputs:
        - res: list, the same size as point_list, contains unicity values
               corresponding to different number of points as side information
    -------
    AF
    """
    res = np.zeros(len(point_list))
    for i, npoints in enumerate(point_list):
        for uid in sampids:
            trace = u2p[uid]
            pset = set(np.random.choice(trace, size=npoints, replace=False))
            res[i] += check_unique(popids, pset, p2u)
    return res


def compute_unicity_series_raw(u2p, p2u, smin, smax, step,
                               sample_size=int(1e4), point_list=[2, 3, 4, 5]):
    """This function is a wrapper for running get_sample_and_pop and
    compute_unicity (defined above) in series and so the docstring is ommitted
    as the variable names are identicle.
    -------
    AF
    """
    alluserids = list(u2p.keys())
    print('getting samples')
    popids, sampids, index = get_sample_and_pop(
        alluserids, smin, smax, step, sample_size)
    print('computing unicity')
    resdf = pd.DataFrame(data=np.zeros(
        shape=(len(popids), len(point_list))), index=index, columns=point_list)
    for i in tq(range(len(index))):
        nunique = compute_unicity(u2p, p2u, popids[i], sampids[i], point_list)
        unicity = nunique / sample_size
        resdf.loc[index[i], :] = unicity
    return resdf
