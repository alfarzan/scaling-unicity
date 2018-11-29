import numpy as np
from scipy import sparse as sps
from dataformat_utils import sparsify_mat_list, vstack_multiply
from dataformat_utils import chunkify_mat_list, get_input_dists
from geoloc_utils import get_geo
from model_source import resampler
from collections import defaultdict
import pandas as pd
from time import time as tm
import random as rnd


from tqdm import tqdm as tq

def get_sample(u2p, sample_size, seed=None):
    """Randomly samples a part of the original u2p matrix.
    
    Inputs:
        - u2p: 5-tuple, output of resampler
        - sample_size: int, specifies the size of population to be sampled
        - seed: int, a seed for the numpy random number generator

    Outputs:
        - sample: 5-tuple similar to output of resampler"""
    
    if seed != None:
        np.random.seed(seed)

    data, rows, cols, shape, rand_acts = u2p
    n, p = shape
    pop = np.random.choice(np.arange(n, dtype=np.int32), size=sample_size, replace=False)
    sacts = rand_acts[pop]

    start_inds = np.cumsum(rand_acts, dtype=np.int32)
    ss = (sample_size, p)  # sample shape

    l = sacts.sum()
    sr = np.ones(l, dtype=np.int32)  # rows
    sc = np.zeros(l, dtype=np.int32)  # cols
    sd = np.ones(l, dtype=np.int8)  # data
    
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
                         pl=[2, 3, 4, 5], cs=int(1e4), seed=None):
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
        - seed: int, seed for both numpy.random.seed and random.seed
    
    Outputs:
        - df: pandas.DataFrame() object which contains the results of the
          calculation. Each row represnts a population size while each column
          represnts the unicity for a particular number of points. It is indexed
          by the population values."""
    if seed != None:
        np.random.seed(seed)
        rnd.seed(seed)

    # getting inputs
    ana = get_geo()

    # generating the first step
    s_u2p = resampler(step, inputs, ana)
    
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

    dur = -1
    print('starting unicity')
    for iii in tq(range(nsteps)):

        # if it's the first one make sure to not regenerate
        if iii != 0:
            u2p = resampler(step, inputs, ana)
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
          points"""
    if seed != None:
        np.random.seed(seed)
    data, rows, cols, shape, rand_acts = sample
    n, p = shape
    smat_list = defaultdict(list)
    for cp in pl:
        smat_list[cp].append(np.ones(n * cp, dtype=np.int8)) # data
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