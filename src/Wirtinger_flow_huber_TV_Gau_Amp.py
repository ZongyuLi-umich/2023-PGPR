# Wirtinger_flow_huber_TV.py
import numpy as np
from utils2 import get_grad_gau_amp, huber, grad_huber, curv_huber, \
                get_sigma2_gau_amp, holocat, diff2d_forw, diff2d_adj
from numpy.linalg import norm
from eval_metric import nrmse
from tqdm import tqdm

        
def Wintinger_flow_huber_TV_Gau_Amp(A, At, y, b, x0, ref, niter, sthow, reg1, reg2, xtrue, verbose = True):
    M = len(y)
    N = len(x0)
    out = []
    sn = np.sqrt(N).astype(int)
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    sqrty = np.sqrt(np.maximum(y - b, 0))
    phi, grad_phi, fisher = get_grad_gau_amp()
    huber_v = np.vectorize(huber)
    grad_huber_v = np.vectorize(grad_huber)
    curv_huber_v = np.vectorize(curv_huber)
    get_sigma2_amp_v = np.vectorize(get_sigma2_gau_amp)
    Ax = A(holocat(x, ref))
    # sigma2 = 2 * np.sum(y) / sn
    sigma2 = np.amax(sqrty) / sn
    # print('sigma2: ', sigma2)
    # sigma2 = get_sigma2_amp_v(Ax, b)*2
    Tx = diff2d_forw(x, sn, sn)
    def cost_fun(Ax, Tx): return np.sum(phi(Ax, sqrty, sigma2)) + reg1 * np.sum(huber_v(Tx, reg2))
    lastnrmse = 1
    for iter in range(niter):
        grad_f = np.real(At(grad_phi(Ax, sqrty, sigma2)))[:N] + reg1 * diff2d_adj(grad_huber_v(Tx, reg2), sn, sn)
        Adk = A(holocat(grad_f, np.zeros_like(grad_f))) # K*L
        Tdk = diff2d_forw(grad_f, sn, sn)
        
        D1 = np.sqrt(fisher(Ax, sigma2))
        D2 = np.sqrt(curv_huber_v(Tx, reg2))
        
        if sthow == 'fisher':
            mu = - (norm(grad_f)**2) / (norm(np.multiply(Adk, D1))**2 + reg1 * (norm(np.multiply(Tdk, D2))**2))
        elif sthow == 'lineser':
            mu = -1
            Ax_old = np.copy(Ax)
            Tx_old = np.copy(Tx)
            cost_old = cost_fun(Ax_old, Tx_old)
            
            Ax_new = Ax + mu * Adk
            Tx_new = Tx + mu * Tdk
            mu_grad_f = 0.01 * norm(grad_f)**2
            
            while cost_fun(Ax_new, Tx_new) > cost_old + mu * mu_grad_f:
                mu /= 2
                Ax_new = Ax + mu * Adk
                Tx_new = Tx + mu * Tdk
        elif sthow == 'empirical':
            mu = -0.00005
        else:
            raise NotImplementedError
        x += mu * grad_f
        x = np.clip(x, 0, 1) # set non-negatives to zero
        Ax = A(holocat(x, ref))
        Tx = diff2d_forw(x, sn, sn)
        out.append(nrmse(x, xtrue))
        
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || nrmse (out, xtrue): {out[-1]:.4f}')
    return x, out

