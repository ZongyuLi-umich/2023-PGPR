# pnp_pgadmm.py
# plug and play ADMM method for Poisson+Gaussian Phase Retrieval
from ast import AugLoad

from utils2 import * 
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from eval_metric import *
import torch

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


def pnp_pgadmm(A, At, y, b, x0, ref, sigma, delta, niter, rho, uiter, mu_u, xtrue, model, verbose=True):
    N = len(x0)
    sn = np.sqrt(N).astype(int)
    out = []
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    u = np.copy(x)
    Au = A(holocat(u, ref))
    eta = np.zeros(N)
    _, grad_phi, fisher = get_grad(sigma, delta)
    
    for iter in range(niter):
        # update u
        for _ in range(uiter):
            grad_u = np.real(At(grad_phi(Au, y, b)))[:N] + rho * (u - x - eta) 
            # Adk = A(holocat(grad_u, np.zeros_like(grad_u)))
            # D1 = np.sqrt(fisher(Au, b))
            #mu_u = - (norm(grad_u)**2)/ (norm(np.multiply(Adk, D1))**2) 
            u = np.maximum(0, u + mu_u * grad_u)
            # update Au
            Au = A(holocat(u, ref))
            
        # update x
        ##########################################
        # Implement plug and play      
        x_tmp = u - eta
        x_tmp = np.clip(x_tmp, 0, 1)
        x_tmp = torch.from_numpy(jreshape(x_tmp, sn, sn))[None, None, ...].to(torch.float32).cuda()
        x_tmp = model(x_tmp)
        x_tmp = vec(x_tmp.cpu().numpy())
        x = np.clip(x_tmp, 0, 1)
        ##########################################
        # update eta
        eta = eta + x - u
        out.append(nrmse(x, xtrue))
        
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || nrmse (u, xtrue): {nrmse(u, xtrue):.4f} || nrmse (out, xtrue): {out[-1]:.4f}')

    return x, out
