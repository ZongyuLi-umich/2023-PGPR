import numpy as np
from utils2 import holocat, get_sigma2_gau_amp
from numpy.linalg import norm
from tqdm import tqdm
import torch
from eval_metric import nrmse
from scipy.sparse.linalg import cg 
from scipy.sparse.linalg import LinearOperator

def Gerchberg_Saxton(A, At, y, b, x0, ref, niter, xtrue, verbose=True):
    M = len(y)
    N = len(x0)
    out = []
    sn = np.sqrt(N).astype(int)
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    sqrty = np.sqrt(np.maximum(y - b, 0))
    Ax = A(holocat(x, ref))
    lastnrmse = 1
    get_sigma2_amp_v = np.vectorize(get_sigma2_gau_amp)
    for iter in range(niter):
        # sigma2 = get_sigma2_amp_v(Ax, b)
        c = np.sign(Ax)
        RHS = np.real(At(sqrty * c))[:N]
        def hessianfunc(x): return np.real(At(A(holocat(x, ref))))[:N]
        LO = LinearOperator((N, N), matvec=hessianfunc, rmatvec = hessianfunc)
        x, _ = cg(LO, RHS)
        
        x = np.clip(x, 0, 1) # set non-negatives to zero
        Ax = A(holocat(x, ref))
        out.append(nrmse(x, xtrue))
        
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || nrmse (out, xtrue): {out[-1]:.4f}')
    return x, out

