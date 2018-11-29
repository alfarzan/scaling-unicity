
import dataformat_utils as dut
import numpy as np
from tqdm import tqdm as tq
from functools import partial
from collections import Counter
import unicity_utils as uut
import pickle


def extract_time(u2p, lhrs, lants):
    time_arr = np.zeros(lhrs)
    get_track = partial(dut.get_user_track, lants=lants)
    for user in range(len(u2p)):
        tx = get_track(u2p[user])
        ti = tx[0]
        time_arr[ti] += 1
    time_arr = time_arr / time_arr.sum()
    return time_arr


def extract_activity(u2p, lhrs):
    actarr = np.zeros(lhrs)
    for user in range(len(u2p)):
        a = len(u2p[user])
        actarr[a] += 1
    actarr = actarr / actarr.sum()
    return actarr


def extract_frequency(u2p, lhrs, lants, sgs=10):
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
    a = extract_activity(u2p, lhrs)
    f = extract_frequency(u2p, lhrs, lants)
    t = extract_time(u2p, lhrs, lants)
    return a, f, t


def get_all_inputs(u2parr, minsamp=1e4, nsamples=20, lhrs=2160, lants=6351):
    # optimised for portugal
    allkeys = np.arange(0, len(u2parr))
    samplist = []
    sampsizes = np.linspace(minsamp, len(u2parr), nsamples, dtype=int)
    for n in sampsizes:
        samp = np.random.choice(allkeys, replace=False, size=n)
        samplist.append(list(samp))

    inputs = []
    for arrs in tq(samplist):
        cu2p = u2parr[arrs]
        inputs.append(get_inputs(cu2p, lhrs, lants))
    return inputs


def gen_unicity_vals(inputs, max_size, step, sample_size=int(1e4),
                     seed=123456):
    dflist = []
    for inp in tq(inputs):
        df = uut.begin_unicity_series(
            max_size, step, sample_size, inp, seed=seed)
        dflist.append(df)
    return dflist


if __name__ == '__main__':
    print('loading the data')
    allu2p = dut.get_u2p(
        root='/data/ali/home/unicity/cdr_tracks/unicity_cdr/cleaned_real_cdr/')
    lhrs = len(dut.get_date_array())
    lants = len(dut.get_ant_array())

    # remapping allu2p to list and then np arrays
    u2parr = list(np.zeros(len(allu2p)))
    for i in tq(allu2p):
        u2parr[i] = allu2p[i]
    u2parr = np.array(u2parr)

    print('getting the inputs')
    inputs = get_all_inputs(u2parr, 20, lhrs=lhrs, lants=lants)

    print('getting the u2ps')
    dflist = gen_unicity_vals(inputs, len(u2parr), int(2e4))
    with open('../temp/learning_curve_data_frame_dict.p', 'wb') as pfile:
        pickle.dump(dflist, pfile)
