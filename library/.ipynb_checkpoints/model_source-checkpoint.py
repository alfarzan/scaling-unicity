import numpy as np
import random as rnd


def gen_cluster(size, ana, ana_keys):
    """This generates a cluster of unique antennas of fixed size.

    Inputs:
        - size: int, indicates size of cluster
        - ana: dict, output of the get_geo function
        - ana_keys: list, the keys of ana which enumerates all antennas present

    Outputs:
        - ndarray of ints which consitutes a connected path of fixed size on
          the antenna network"""
    start = rnd.sample(ana_keys, 1)[0]
    current_ant = start
    visited = {current_ant}
    choices = set()
    while len(visited) != size:
        choices = choices.union(ana[current_ant]) - visited
        if len(choices) == 0:  # if this happens then restart
            current_ant = rnd.sample(ana_keys, 1)[0]
            visited = {current_ant}
            choices = choices.union(ana[current_ant]) - visited
        current_ant = rnd.sample(choices, 1)[0]
        visited.add(current_ant)
    v = list(visited)
    v = np.array(v, dtype=np.int32)
    np.random.shuffle(v)
    return v


def create_cluster_array(nusers, size, ana):
    """Generates a 'nusers' clusters of fixed size on the antenna network

    Inputs:
        - nusers: int, number of clusters
        - size: int, size of each cluster
        - ana: dict, output of get_geo

    Outputs:
        - ndarray of shape (nusers, size)"""
    antlist = list(ana.keys())
    arr = np.zeros((nusers, size), dtype=np.int32)
    for i in range(nusers):
        arr[i] = gen_cluster(size, ana, antlist)
    return arr


def resampler_non_sparse_matrix(nusers, cluster_array, input_dists, ana):
    """Generates the synthetic data and stores data in dictionary.
    See resampler docstring for more info."""
    assert len(cluster_array) >= nusers
    # unpacking the input distributions
    act, fbar, time = input_dists
    n = len(ana)
    # antlist = list(ana.keys())
    hrs = np.arange(len(time))
    acts = np.arange(cluster_array.shape[1], cluster_array.shape[1] + len(act))
    u2p = {}
    for user in range(nusers):
        a = np.random.choice(acts, p=act)
        t = np.random.choice(hrs, size=a, p=time, replace=False)
        s = cluster_array[user]
        x = np.random.choice(s, a, p=fbar)
        u2p[user] = t * n + x
    return u2p


def resampler(nusers, cluster_array, inputs, ana):
    """
    Generates synthetic data using the input arrays, the antenna network and
    pre-computed antenna clusters. Returns the information such that it can be
    passed to scipy.sparse.csr_matrix((data, (row, col), shape)).

    Inputs:
        - nusers: int, indicates the number of users to be generated
        - cluster_array: ndarray containing pre-computed clusters (see
          create_cluster_array() for more info)
        - inputs: 3-tuple of ndarrays, contains the activity, frequency and
          circadian distibutions
        - ana: dict, used only to enumerate over antennas (see get_geo())

    Outputs:
        - data: ndarray of ones. Indicates values of the sparse matrix
        - rows: ndarray, indicates row indices of sparse matrix
        - cols: ndarray, indicates column indices of sparse matrix
        - rand_acts: ndarray, contains activity values for all users present
        - shape: 2-tuples, contains shape of sparse matrix
    """
    assert len(cluster_array) >= nusers
    # unpacking the input distributions
    act, fbar, time = inputs
    n = len(ana)
    # antlist = list(ana.keys())
    hrs = np.arange(len(time), dtype=np.int32)
    acts = np.arange(cluster_array.shape[1],
                     cluster_array.shape[1] + len(act), dtype=np.int32)

    p = n * len(time)
    shape = (nusers, p)

    rand_acts = np.random.choice(acts, size=nusers, p=act)
    nnz = rand_acts.sum()
    rows, cols = np.ones(nnz, dtype=np.int32), np.zeros(nnz, dtype=np.int32)
    data = np.ones(nnz, dtype=np.int8)

    currind = 0
    for user in range(nusers):
        a = rand_acts[user]
        t = np.random.choice(hrs, size=a, p=time, replace=False)
        s = cluster_array[user]
        x = np.random.choice(s, a, p=fbar)
        cols[currind: currind + a] = t * n + x
        rows[currind: currind + a] = user * rows[currind: currind + a]
        currind += a
    return data, rows, cols, shape, rand_acts
