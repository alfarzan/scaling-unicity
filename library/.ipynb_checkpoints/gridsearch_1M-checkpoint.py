"""
This script runs the unicity model using various different
distribution parameter values. For 1M people.
"""

from unicity_utils import begin_unicity_series
from dataformat_utils import get_pool_data, gen_act, get_input_dists
import numpy as np
import random as rnd
import multiprocessing as mp
import os
import pickle
from tqdm import tqdm as tq
from scipy.special import zeta


def gen_freq(pars, sgs=10):
    def frequency_fit(x, a):
        return pow(x, -a) / zeta(a, 1)
    x = np.arange(1, sgs + 1)
    if type(pars) == tuple:
        return frequency_fit(x, *pars)
    else:
        return frequency_fit(x, pars)


def worker(params):
    _, _, circ = get_input_dists()
    nhrs = len(circ)
    (fpars, apars), bnorm = params[3]
    act = gen_act(apars, nhrs, bnorm)
    f = gen_freq(fpars, params[6])
    act = act / act.sum()
    f = f / f.sum()
    inputs = (act, f, circ)
    params[3] = inputs
    df = begin_unicity_series(*params)
    df.to_csv('../results/gridsearch_1M_fine_2/iter_{}.csv'.format(params[7]))
    return(1)


def instantiate_pool(allpars, max_size, step, sample_size, pl, cs, sgs):
    nproc = len(allpars[0])
    nproc = min(nproc, 36)
    data = get_pool_data(max_size, step, allpars, sample_size, pl, cs, sgs)
    print('begining multiprocessed pool:')
    print('max_size:      {}'.format(max_size))
    print('step:          {}'.format(step))
    print('sample_size:   {}'.format(sample_size))
    print('point list:    {}'.format(pl))
    print('cluster_size:  {}'.format(cs))
    print('subgraph size: {}'.format(sgs))
    print('nproc:         {}'.format(nproc))
    print('# of params:   {}'.format(len(allpars[0])))
    directory = '../results/gridsearch_1M_fine_2/'.format(max_size // 1e6)
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

    # setting seed
    rnd.seed(1370)
    np.random.seed(1370)

    max_size = int(1e6)
    step = int(2e4)
    sample_size = int(1e4)
    pl = [2, 3, 4, 5]
    cs = int(2e4)
    sgs = 10

    # created in fiiting_forms.ipynb
    with open('../inputs/fine_1m_gridsearch_2.p', 'rb') as gsp:
        allpars = pickle.load(gsp)

    instantiate_pool(allpars, max_size, step, sample_size, pl, cs, sgs)
