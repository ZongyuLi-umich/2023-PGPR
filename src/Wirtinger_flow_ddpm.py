# Wirtinger_flow_ddpm.py
import numpy as np
from utils2 import *
from numpy.linalg import norm
from tqdm import tqdm
import torch
import math
from eval_metric import *


def grad_gau(v, yi, bi): return 4 * (abs2(v) + bi - yi) * v
def grad_pois(v, yi, bi): return 2 * v * (1 - yi / (abs2(v) + bi))
        
def Wirtinger_flow_score_ddpm(A, At, y, b, x0, ref, sigma, delta, 
                                niter, xtrue, model, gradhow = 'pois',
                                verbose=True):
    M = len(y)
    N = len(x0)
    out = []
    sn = np.sqrt(N).astype(int)
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    phi, grad_phi, fisher = get_grad(sigma, delta)

    Ax = A(holocat(x, ref))
    lastnrmse = 1
    
    grad_gau_v = np.vectorize(grad_gau)
    grad_pois_v = np.vectorize(grad_pois)

    totalbetas = 100
    betas = np.linspace(1e-4, 0.3, num=totalbetas)
    alphas = 1-betas
    alphabars = np.cumprod(alphas)

    negone = True

    for t in range(niter, 2, -1):
        lsize = 128
        if negone:
            x = x*2-1
        netinput = torch.from_numpy(np.reshape(x, (1,1,lsize,lsize))).float().cuda()
        network_t = torch.from_numpy(np.array([t])).float().cuda()
        epspart = model.forward(netinput, network_t).cpu().detach().numpy().reshape((lsize, lsize))
        epspart = epspart.reshape(-1)
        #scorepart = -model.forward(torch.from_numpy(x.reshape((lsize, lsize)))).cpu().detach().numpy()/0.05
        epspart = np.squeeze(epspart)

        if t > 1:
            z = np.random.randn(lsize**2)
        else:
            z = np.zeros(lsize**2)
        sigma_t = 0.01
        # sigma_t = math.sqrt(betas[t-1]) #one possible choice

        x = (x - (1-alphas[t-1])/math.sqrt(1-alphabars[t-1]) * epspart)/math.sqrt(alphas[t-1]) + sigma_t * z
        if negone:
            x = (x+1)/2
        #x = x + (sigmas[iter-1]**2 - sigmas[iter]**2) * scorepart
        if gradhow == 'gau':
            grad_f = np.real(At(grad_gau_v(Ax, y, b)))[:N] 
        elif gradhow == 'pois':
            grad_f = np.real(At(grad_pois_v(Ax, y, b)))[:N] 
        elif gradhow == 'pg':
            grad_f = np.real(At(grad_phi(Ax, y, b)))[:N]
        else:
            raise NotImplementedError
        
        D1 = np.sqrt(fisher(Ax, b))
        Adk = A(holocat(grad_f, np.zeros_like(grad_f)))
        mu = -(norm(grad_f)**2) / (norm(np.multiply(Adk, D1))**2)
        # update x
        x += mu * grad_f
        x = np.clip(x, 0, 1) # set non-negatives to zero
        # update Ax
        Ax = A(holocat(x, ref))

        out.append(nrmse(x, xtrue))
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        if verbose: 
            print(f'iter: {niter-t:03d} / {niter:03d} || nrmse (out, xtrue): {out[-1]:.4f}')
    return x, out

