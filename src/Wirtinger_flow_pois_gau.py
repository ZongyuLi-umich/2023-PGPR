# Wirtinger_flow_huber_TV.py
import numpy as np
from utils2 import *
from numpy.linalg import norm
from tqdm import tqdm
from eval_metric import *

def get_grad(sigma, delta):
    def phi(v, yi, bi): return (abs2(v) + bi) - yi * np.log(abs2(v) + bi)
    def grad_phi(v, yi, bi): 
        if yi < 100:
            u = abs2(v) + bi
            return 2 * v * grad_phi1(u, yi, sigma, delta)
        else:
            return 2 * v * (1 - yi / (abs2(v) + bi))
    def fisher(vi, bi): return 4 * abs2(vi) / (abs2(vi) + bi)
    return np.vectorize(phi), np.vectorize(grad_phi), np.vectorize(fisher)
        
def Wintinger_flow_pois_gau(A, At, y, b, x0, ref, sigma, delta, 
                            niter, reg1, reg2, xtrue):
    M = len(y)
    N = len(x0)
    out = []
    sn = np.sqrt(N).astype(int)
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    phi, grad_phi, fisher = get_grad(sigma, delta)
    huber_v = np.vectorize(huber)
    grad_huber_v = np.vectorize(grad_huber)
    curv_huber_v = np.vectorize(curv_huber)
    Ax = A(holocat(x, ref))
    Tx = diff2d_forw(x, sn, sn)
    def cost_fun(Ax, Tx): return np.sum(phi(Ax, y, b)) + reg1 * np.sum(huber_v(Tx, reg2))
    for iter in tqdm(range(niter)):
        grad_f = np.real(At(grad_phi(Ax, y, b)))[:N] + reg1 * diff2d_adj(grad_huber_v(Tx, reg2), sn, sn)
        Adk = A(holocat(grad_f, np.zeros_like(grad_f))) # K*L
        Tdk = diff2d_forw(grad_f, sn, sn)
        
        D1 = np.sqrt(fisher(Ax, b))
        D2 = np.sqrt(curv_huber_v(Tx, reg2))
        mu = - (norm(grad_f)**2) / (norm(np.multiply(Adk, D1))**2 + reg1 * (norm(np.multiply(Tdk, D2))**2))
        x += mu * grad_f
        x[(x < 0)] = 0 # set non-negatives to zero
        Ax = A(holocat(x, ref))
        Tx = diff2d_forw(x, sn, sn)
        out.append(nrmse(x, xtrue))
    return x, out

