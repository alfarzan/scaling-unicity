"""
This file computes the range of parameters for the beta and power law functions
according to the earth movers' distance (EMD) of the resulting distributions
from the empirical distribution.

Author: Ali Farzanehfar
"""


import numpy as np
import scipy.optimize as so
import scipy.stats as scp
from dataformat_utils import activity_fit, frequency_fit
from tqdm import tqdm as tq


def gen_freq_params(emthresh=0.7, nvals=8, sgs=10):
    """
    This function computes the range of frequency parameters to use in order to
    maintain a distance of less than 0.7 EMD compared to the maximum possible
    distance given the functional form of a power law.

    Inputs:
        - emthresh: float, the upper limit on the most divergent distributions
                    from the emprical fit as a fraction of the maximum distance
        - nvals: int, number of frequency distritbution parameters to return
        - sgs: int, size of the distribution fit (important for normalisation)

    Outputs:
        - freq_params: list, a list of the parameters for the powerlaw fit
    -------
    AF
    """

    fbar = np.load('../inputs/frequency.npy')
    fbar = fbar[:sgs] / np.sum(fbar[:sgs])

    x = np.arange(1, sgs + 1)

    fitted_freq_params, covmat = so.curve_fit(frequency_fit, x, fbar)
    fit_f = frequency_fit(x, fitted_freq_params[0], fitted_freq_params[1])

    alpha_range = np.linspace(0, 5, 1000)

    em = []
    for alpha in alpha_range:
        curr_f = pow(x, -alpha)
        const = 1 / np.sum(curr_f)
        em.append(scp.wasserstein_distance(curr_f * const, fit_f))

    em = np.array(em)
    em = em / em.max()
    idx = np.argmin(em)

    lrange = alpha_range[np.argmin(abs(em[:idx] - emthresh))]
    hrange = alpha_range[idx + np.argmin(abs(em[idx:] - emthresh))]

    calculated_consts = []
    alrange = np.linspace(lrange, hrange, nvals)
    for alpha in alrange:
        curr_f = pow(x, -alpha)
        const = 1 / np.sum(curr_f)
        calculated_consts.append(const)

    freq_params = list(zip(alrange, calculated_consts))
    return freq_params


def gen_act_params(emthresh=0.7, nvals=9):
    """
    This function does the same as the above function for the activity
    distribution and so the docstring is ommited.
    -------
    AF
    """

    act = np.load('../inputs/activity.npy')
    act = act / act.max()
    x = np.linspace(0, 1, len(act))

    def mse(a, b):
        res = (1 / len(a)) * sum(pow(a - b, 2))
        return res

    arange = np.linspace(0.5, 10, 100)
    brange = np.linspace(0.5, 13, 100)
    crange = np.linspace(1e-6, 10, 100)

    mse_list = []
    parvals = []
    for ai in tq(arange):
        for bi in brange:
            for ci in crange:
                currbeta = activity_fit(x, ai, bi, ci)
                parvals.append((ai, bi, ci))
                mse_list.append(mse(act, currbeta))

    inters = 500
    a_inter = np.linspace(1, 2.5, inters)
    b_inter = np.linspace(2, 30, inters)

    avals, bvals, em = [], [], []

    actfit = activity_fit(x, *parvals[np.argmin(mse_list)])
    actfit = actfit / actfit.sum()

    avals = []
    bvals = []
    for ai in tq(a_inter):
        for bi in b_inter:
            curr_a = activity_fit(x, ai, bi, 1)
            const = 1 / np.sum(curr_a)
            em.append(scp.wasserstein_distance(curr_a * const, actfit))
            avals.append(ai)
            bvals.append(bi)

    em = np.array(em)
    em = em / em.max()
    idx = np.argmin(em)
    avals = np.array(avals)
    bvals = np.array(bvals)

    larange = avals[np.argmin(abs(em[:idx] - emthresh))]
    harange = avals[idx + np.argmin(abs(em[idx:] - emthresh))]

    lbrange = bvals[np.argmin(abs(em[:idx] - emthresh))]
    hbrange = bvals[idx + np.argmin(abs(em[idx:] - emthresh))]

    alrange = np.linspace(larange, harange, int(pow(nvals, 0.5)))
    blrange = np.linspace(lbrange, hbrange, int(pow(nvals, 0.5)))

    params = []
    for ai in alrange:
        for bi in blrange:
            cb = activity_fit(x, ai, bi, 1)
            ci = 1 / cb.sum()
            params.append((ai, bi, ci))
    return params


def gen_input_dist_params(ef, nf, ea, na):
    """
    A wrapper for the above two functions to generate the parameters given a
    certain threshold and a certain number of parameters for both the activity
    and the frequency distributions.

    Inputs:
        - ef: float, threshold for frequency (between 0 and 1)
        - ea: float, threshold for activity (between 0 and 1)
        - na: int, number of activity distributions to be generated
        - nf: int, number of frequency distributions to be generated

    Outputs:
        - combos: list, containing all the possible instatiations fo the
                  frequency and activity distributions generated.
    -------
    AF
    """
    f = gen_freq_params(ef, nf)
    a = gen_act_params(ea, na)
    combos = []
    for i in f:
        for j in a:
            combos.append((i, j))
    return combos


def wrapped_gen_dist(em, nf, na):
    """
    A wrapper for the gen_input_dist_params function including a check to make
    sure the number of activity parameters checked is a full square so that
    each of the two beta function parameters is checked for an equal number of
    different values.
    -------
    AF
    """
    assert pow(na, 0.5) % 1 == 0  # making sure na is a full square
    return gen_input_dist_params(em, nf, em, na)
