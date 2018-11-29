"""
This script runs the unicity model using various different
distribution parameter values.

Author: Ali Farzanehfar
"""

from unicity_utils import begin_unicity_series
from dataformat_utils import get_pool_data, gen_act, gen_freq, get_input_dists
import numpy as np
import random as rnd
import multiprocessing as mp
import os
from tqdm import tqdm as tq
from generate_gridsearch_params import wrapped_gen_dist


def worker(params):
    print('Instantiating unicity worker {}'.format(params[-1]))
    _, _, circ = get_input_dists()
    nhrs = len(circ)
    fpars, apars = params[3]
    act = gen_act(apars, nhrs)
    f = gen_freq(fpars, params[6])
    act = act / act.sum()
    f = f / f.sum()
    inputs = (act, f, circ)
    params[3] = inputs
    df = begin_unicity_series(*params)
    nmils = params[0] // 1e6
    df.to_csv(
        '../results/gridsearch_{:.0f}M/iter_{}.csv'.format(nmils, params[7]))


def instantiate_pool(allpars, max_size, step, sample_size, pl, cs, sgs,
                     max_nprc):
    nproc = min(max_nprc, len(allpars))
    data = get_pool_data(max_size, step, allpars, sample_size, pl, cs, sgs)
    print('begining multiprocessed pool:')
    print('Processes:     {}'.format(nproc))
    print('max_size:      {}'.format(max_size))
    print('step:          {}'.format(step))
    print('sample_size:   {}'.format(sample_size))
    print('point list:    {}'.format(pl))
    print('cluster_size:  {}'.format(cs))
    print('subgraph size: {}'.format(sgs))
    nmils = max_size // 1e6
    directory = '../results/gridsearch_{:.0f}M/'.format(nmils)
    if not os.path.exists(directory):
        os.makedirs(directory)

    mypool = mp.Pool(nproc)
    jobs = []
    for elem in data:
        jobs.append(mypool.apply_async(worker, args=(elem,)))

    mypool.close()

    for proc in tq(jobs):
        proc.get()

    mypool.join()


if __name__ == '__main__':
    emthresh = 0.7
    total = 4 ** 3
    allpars = wrapped_gen_dist(emthresh, total)
    assert total == len(allpars)

    # setting seed
    rnd.seed(1370)
    np.random.seed(1370)

    max_size = int(6e7)
    step = int(5e5)
    sample_size = int(1e4)
    pl = [2, 3, 4, 5]
    cs = int(1e5)
    sgs = 10
    max_nprc = 12
    # # created in fiiting_forms.ipynb
    # with open('../inputs/gridsearch_params_1M.p', 'rb') as gsp:
    #     allpars = pickle.load(gsp)

    instantiate_pool(allpars, max_size, step,
                     sample_size, pl, cs, sgs, max_nprc)
