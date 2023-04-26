# pnp_pgadmm.py
# plug and play ADMM method for Poisson+Gaussian Phase Retrieval
from ast import AugLoad

from utils2 import * 
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from eval_metric import *
import torch


def pnp_pgprox(A, At, y, b, x0, ref, sigma, delta, niter, xtrue, model, mu = None, scale = 1, rho=1, verbose=True):
    
    # xiaojian
    # x0 = denoise(x0, model, scale, sn=128)
    
    N = len(x0)
    sn = np.sqrt(N).astype(int)
    out = []
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    Ax = A(holocat(x, ref))
    _, grad_phi, fisher = get_grad(sigma, delta)
    lastnrmse = 1
    
    for iter in range(niter):
        grad_x = np.real(At(grad_phi(Ax, y, b)))[:N]
        if mu is None:
            Adk = A(holocat(grad_x, np.zeros_like(grad_x)))
            D1 = np.sqrt(fisher(Ax, b))
            mu = - (norm(grad_x)**2)/ (norm(np.multiply(Adk, D1))**2) 
        
        x1 = np.maximum(0, x + mu * grad_x)
        x2 = denoise(x1, model, scale, sn=128)  
        x = (1-rho) * x1 + rho * x2
        x = np.clip(x, 0, 1)
              
        Ax = A(holocat(x, ref))
        out.append(nrmse(x, xtrue))
        
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || scale: {scale:.2f} || step: {mu:.2e} || nrmse (x1, xtrue): {nrmse(x1, xtrue):.4f} || nrmse (x2, xtrue): {nrmse(x2, xtrue):.4f} || nrmse (out, xtrue): {out[-1]:.4f}')

    return x, out


def denoise(x_tmp, model, scale, sn=128):
    with torch.no_grad():
        x_tmp = np.clip(x_tmp, 0, 1)
        x_tmp = torch.from_numpy(jreshape(x_tmp, sn, sn))[None, None, ...].to(torch.float32).cuda()
        x_tmp = model(x_tmp * scale)/scale
        x_tmp = vec(x_tmp.cpu().numpy())
        x_tmp = np.clip(x_tmp, 0, 1)
        return x_tmp
    
