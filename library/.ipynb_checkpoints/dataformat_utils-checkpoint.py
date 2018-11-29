"""
This file provides a series of helper function to use for unicity computations

Author: Ali Farzanehfar
"""

import numpy as np
from scipy import sparse as sps
import pandas as pd
import os
from tqdm import tqdm as tq
import pickle
import datetime
from collections import defaultdict


def get_u2p(rootdir, start_date='2007-01-15', end_date='2007-04-15',
            max_pop=False, uselist=False):
    """
    This function loads up pre-processed pickled arrays in pre-specified date
    ranges.
    It returns a list of numpy arrays representing the sparse user tracks.

    Unfortnately the data is not publicly shareable due to privacy concerns.
    This function is included for transparency.

    Each pickled array would contain a varying number of trajectories, stored
    as lists of integers. Each integer corresponding to a space-time point.

    NOTE: noramlly get_u2p has default values but here I have removed them. You
    should put them back for your specific application to ensure every other
    file runs smoothly.
    -------
    AF
    """
    narrs = os.listdir(rootdir)
    names = list(map(lambda x: rootdir + x, narrs))
    a = []
    print('loading arrays')
    for name in tq(names):
        a += pickle.load(open(name, 'rb'))

    if uselist:
        return a

    del narrs
    del names

    if max_pop is not False:
        np.random.shuffle(a)
        a = a[:max_pop]
    a = {i: a[i] for i in range(len(a))}
    return a


def get_p2u(u2p):
    """
    Takes in a user to points dictionary and constructs a point to users
    dictionary.

    Inputs:
        - u2p: dict, a dictonary of the form {uid: set(points)}

    Outputs:
        - p2u: dict, a complement dictionary of the form {point: set(uids)}
    -------
    AF
    """
    p2u = defaultdict(list)
    for uid in tq(u2p):
        for point in u2p[uid]:
            p2u[point].append(uid)
    return dict(p2u)


def get_user_track(indices, lants):
    """
    This takes the condensed representation of user tracks and returns the
    detangled (time/ antenna) representation.

    Note that this will only work if the ordering of the antennas is not
    altered throughout the whole process (which is the case unless explicitly
    altered)

    Inputs:
        - indices: list, integers representing a particular trajectory
        - lants: int, the total number of antennas

    Outputs:
        - t: numpy array, representing the times in the trace
        - x: numpy array, representing the locations in the trace
    -------
    AF
    """
    a = len(indices)
    x = np.zeros(a, dtype=np.int32)
    t = np.zeros(a, dtype=np.int32)
    for i, ind in enumerate(indices):
        # This is a tuple that correponds to (time, antenna)
        t[i] = ind // lants
        x[i] = ind % lants
    return t, x


def generate_user_indices(tx, lants):
    """
    Reverses get_user_track where t, x = tx
    and tx is a 2-tuple of numpy arrays of ints
    -------
    AF
    """
    t, x = tx
    return (t * lants) + x


def get_date_array(start_date='2007-01-15', end_date='2007-04-15'):
    """
    Takes in strings of the format YYY-MM-DD for start and end dates.

    Then constructs, using the datetime library, a list of strings
    which corresponds to hourly increments from start date (incl)
    to end date(excl).
    -------
    AF
    """
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    step = datetime.timedelta(seconds=3600)
    current = start
    date_list = []
    while current < end:
        date_list.append(current.strftime('%Y-%m-%d %H'))
        current += step
    return np.asarray(date_list)


def get_ant_array(fpath):
    """
    This function simply gets a list of antenna string numbers as they appear
    in the source file located at fpath (str).

    The reason I load this everytime is to ensure that the order of antennas
    is perserved.
    -------
    AF
    """
    adict = []
    with open(fpath, 'r') as fadd:
        for line in fadd:
            ant = line.strip().split(' ')[0]
            adict.append(ant)
    return np.asarray(adict)


def activity_fit(x, a, b, c):
    """
    A beta function for fitting activity distributions
    -------
    AF
    """
    return c * pow(x, a - 1) * pow(1 - x, b - 1)


def gen_act(pars, nhrs):
    """
    A function to generate the activity distribution from the beta function
    -------
    AF
    """
    x = np.linspace(0, 1, nhrs)
    return activity_fit(x, *pars)


def frequency_fit(ant_rank, exp, const):
    """
    The frequency power law function for fitting frequency distributions
    -------
    AF
    """
    return const * pow(ant_rank, -exp)


def gen_freq(pars, sgs=10):
    """
    A function to generate the frequency distribution from the power law
    -------
    AF
    """
    x = np.arange(1, sgs + 1)
    return frequency_fit(x, *pars)


def get_pool_data(max_size, step, dist_pars, sample_size=int(1e4),
                  pl=[2, 3, 4, 5], cs=int(1e4), sgs=10):
    """
    This function is used for parallelizing the unicity computation. The output
    of this is fed into a function which is then run on multiple cores. Can be
    ignored for research purposes.
    -------
    AF
    """
    data = []
    for i in range(len(dist_pars)):
        data.append([max_size, step, sample_size,
                     dist_pars[i], pl, cs, sgs, i + 1])
    return data


def sparsify_mat_list(mat_list):
    """Takes a list of inputs that can feed into
    scipy.sparse.csr_matrix((data, (row, col), shape)) and returns a list of
    scipy.sparse.csr_matrix() objects.

    Inputs:
        - mat_list: list, see get_unicity_subpop() docstring

    Outputs:
        - res: list of scipy.sparse.csr_matrix() objects
    -------
    AF
    """
    res = []
    for mat in mat_list:
        data, rows, cols, shape, _ = mat
        res.append(sps.csr_matrix((data, (rows, cols)), shape=shape))
    return res


def vstack_multiply(res, sample_dict):
    """Takes in a list of sparse matrices and a dict of sparse matrices. Returns
    a dictionary with the same keys of 'sample_dict' where each entry is the
    vstacked multiplication such that the results are equal to the case where
    all matices in the matrix list were vstacked and multiplied by the other
    matrix.

    Inputs:
        - res: list of scipy.sparse.csr_matrix()
        - sample_dict: dict of scipy.sparse.csr_matrix() objects where keys are
          number of points

    Outputs:
        - ps: dict of scipy.sparse.csr_matrix() which is the dot product of all
          the res with sample dict
    -------
    AF
    """
    ps = {}
    for point in sample_dict:
        s = sample_dict[point]
        ps_list = []
        for smat in res:
            ps_list.append(smat.dot(s))
        ps[point] = sps.vstack(ps_list)
    return ps


def chunkify_mat_list(u2p, cs):
    """splits up u2p (returned by resampler) into parts of cs size and returns
    them in a list to be converted to sparse matrices.

    Inputs:
        - u2p: 5-tuple, output of resampler
        - cs: int, representing the number of users in each chunk

    Outputs:
        - mat_list: list of inputs to be fed into scipy.sparse.csr_matrix.
    -------
    AF
    """

    data, rows, cols, shape, rand_acts = u2p
    n, p = shape

    # make sure the matrices are full
    assert n % cs == 0

    start_col_inds = np.cumsum(rand_acts, dtype=np.int32)
    nchunks = n // cs

    cshape = (cs, p)

    mat_list = []
    cchunk = 0
    for chunk in range(nchunks):
        cacts = rand_acts[cchunk:cchunk + cs]
        nnz = cacts.sum()

        crows = np.ones(nnz, dtype=np.int32)
        ccols = np.zeros(nnz, dtype=np.int32)
        cdata = np.ones(nnz, dtype=np.int8)
        mat_list.append((cdata, crows, ccols, cshape, cacts))

        cind = 0
        for user in range(cs):
            ind = cchunk + user
            a = cacts[user]
            crows[cind:cind + a] = user * crows[cind:cind + a]
            start = start_col_inds[ind]
            ccols[cind:cind + a] = cols[start - a:start]
            cind += a
        cchunk += cs
    return mat_list


def get_input_dists(size=10, fnames, inputdir):
    """Loads and normalises input arrays for synthetic data generation.

    Inputs:
        - size: int, cluster size. Used to normalise frequency vector.
        - fnames: list of size 3 containing strings which specify the name of
          saved numpy arrays containing the input distributions.
        - inputdir: str, name of directory containings the saved numpy arrays

    Outputs:
        - a: ndarray, contains the activity distribution
        - f: ndarray, contains the frequency distribution
        - c: ndarray, contains the circadian distribution of activity
    -------
    AF
    """
    # loading the input distributions
    a, c, f = fnames
    a = np.load(inputdir + a)
    c = np.load(inputdir + c)
    f = np.load(inputdir + f)

    # preparing the dists
    f = f[:size] / np.sum(f[:size])
    a = a / np.sum(a)
    c = c / np.sum(c)
    return a, f, c
