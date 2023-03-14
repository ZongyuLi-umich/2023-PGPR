# eval_metric.py
# evaluation metric
import numpy as np
from utils import vec
from numpy.linalg import norm

def phase_shift(x, xtrue): 
    if not np.any(x):
        return 1
    else:
        return np.sign(np.dot(vec(xtrue), x))
def nrmse(x, xtrue): 
    return norm(x - vec(xtrue) * phase_shift(x, xtrue)) / \
                    norm(vec(xtrue) * phase_shift(x, xtrue))