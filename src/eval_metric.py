# eval_metric.py
# evaluation metric
import numpy as np
from utils2 import vec, jreshape
from numpy.linalg import norm
from skimage.metrics import structural_similarity


def phase_shift(x, xtrue): 
    if not np.any(x):
        return 1
    else:
        return np.sign(np.dot(vec(xtrue), vec(x)))
    
def nrmse(x, xtrue): 
    return norm(vec(x) - vec(xtrue) * phase_shift(x, xtrue)) / \
                    norm(vec(xtrue) * phase_shift(x, xtrue))

def ssim(x, xtrue, nx):
    x = 0.5 + x
    xtrue = 0.5 + xtrue
    score = structural_similarity(x, jreshape(vec(xtrue) * phase_shift(x, xtrue), nx, nx), 
                                          data_range=1.0)
    return score