"""
This script computes the unicity values using the model up to a specified
population size

Author: Ali Farzanehfar
"""

from unicity_utils import begin_unicity_series
from dataformat_utils import get_input_dists


max_size = int(6e7)
step = int(5e5)
sample_size = int(1e4)
sgs = 10
cs = int(1e5)
seed = 1038
pl = [2, 3, 4, 5]


inputs = get_input_dists(10, ['activity.npy', 'circadian.npy', 'frequency.npy'], '../inputs/')
df = begin_unicity_series(max_size, step, sample_size,
                          inputs, pl, cs, sgs, seed,
                          verbose=True, autosave='../tmp')

df.to_csv('../results/60M_model.csv')
