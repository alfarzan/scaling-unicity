"""
This file extracts the frequency vectory for each user in the data and puts
the values in a histogram
"""
import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
from scipy.optimize import curve_fit
from numba import jit
from scipy.special import zeta
from scipy.stats import chisquare

@jit(nopython=True)
def counter(a):
    unique = list(set(a))
    unique = np.array(unique)
    counts = np.zeros(len(unique))
    for elem in unique:
        n_elem = len(np.where(a == elem)[0])
        ind = np.where(unique == elem)[0][0]
        counts[ind] = n_elem
    unique_sorter = (-counts).argsort()
    unique = unique[unique_sorter]
    counts = counts[unique_sorter]
    return unique, counts


@jit(nopython=True)
def count_unique(a):
    return len(set(a))

def power_law(n, exp):
    return pow(n, -exp) / zeta(exp, 1)

u2p = dut.get_u2p()

lhrs = len(dut.get_date_array())
lants = len(dut.get_ant_array())

mean_f = np.zeros(lhrs)
powers = np.zeros(len(u2p))
chi2 = np.zeros(len(u2p))
n = np.arange(1, 1000)
diff = np.zeros(lhrs)


i = 0
for user in tq(u2p):
    t, x = dut.get_user_track(u2p[user], lants)
    c = count_unique(x)
    _, fi = counter(x)
    fi = fi / np.sum(fi)
    exp = curve_fit(power_law, np.arange(1, c + 1), fi)[0][0]
    powers[i] = exp
    fit = power_law(n[:c], exp)
    chi2[i] = chisquare(fi, fit)[0]
    fi = np.pad(fi, (0, lhrs - c), 'constant')
    fit = np.pad(fit, (0, lhrs - c), 'constant')
    mean_f += fi
    diff += abs(fi - fit)
    i += 1
    
np.save('/data/ali/home/unicity_scale/inputs/freq_fit_difference.npy', diff)
np.save('/data/ali/home/unicity_scale/inputs/freq_fit_chi2.npy', chi2)
np.save('/data/ali/home/unicity_scale/inputs/frequency.npy', mean_f)
np.save('/data/ali/home/unicity_scale/inputs/freq_fit_exponents.npy', powers)
