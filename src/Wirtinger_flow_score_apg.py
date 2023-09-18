# Wirtinger_flow_score_apg.py
import numpy as np
from utils2 import *
from numpy.linalg import norm
from tqdm import tqdm
import torch
from eval_metric import *


def grad_gau(v, yi, bi): return 4 * (abs2(v) + bi - yi) * v
def grad_pois(v, yi, bi): return 2 * v * (1 - yi / (abs2(v) + bi))
def cost_gau(v, yi, bi): return abs2(abs2(v)+bi-yi)
def cost_pois(v, yi, bi): return (abs2(v) + bi) - yi * np.log(abs2(v) + bi)
        
def Wintinger_flow_score_apg(A, At, y, b, x0, ref, sigma, delta, 
                            niter, xtrue, model, gradhow = 'pois',
                            verbose=True):
    M = len(y)
    N = len(x0)
    out = []
    sn = np.sqrt(N).astype(int)
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0) # primary sequence
    z = np.copy(x) # secondary sequence
    w = np.copy(x) # secondary sequence
    vv = np.copy(x) # secondary sequence
    
    phi, grad_phi, fisher = get_grad(sigma, delta)

    Ax = A(holocat(x, ref))
    
    lastnrmse = 1
    T = 2  # set T=2 for purple dataset
    gamma = 0.5
    sigmas = np.geomspace(0.094, 0.0026, niter)
    tn = 1 # OGM
    x_old = np.copy(x)
    tn_old = np.copy(tn)
    grad_gau_v = np.vectorize(grad_gau)
    grad_pois_v = np.vectorize(grad_pois)
    cost_gau_v = np.vectorize(cost_gau)
    cost_pois_v = np.vectorize(cost_pois)
    if gradhow == 'gau':
        grad_fun = grad_gau_v
        cost_fun = cost_gau_v
    elif gradhow == 'pois':
        grad_fun = grad_pois_v
        cost_fun = cost_pois_v
    elif gradhow == 'pg':
        grad_fun = grad_phi
        cost_fun = phi
    else:
        raise NotImplementedError
    # sigmas = np.geomspace(0.04, 0.0069, niter)
    for iter in range(niter):
        lsize = 128
        for t in range(T):
            network_sigma = torch.from_numpy(np.array([sigmas[iter]])).float().to(next(model.parameters()).device)
            netinput_x = torch.from_numpy(np.reshape(x, (1,1,lsize,lsize))).float().to(next(model.parameters()).device)
            scorepart_x = -model.forward(netinput_x, network_sigma).cpu().detach().numpy().reshape((lsize, lsize))/sigmas[iter]
            scorepart_x = scorepart_x.reshape(-1)
            #scorepart = -model.forward(torch.from_numpy(x.reshape((lsize, lsize)))).cpu().detach().numpy()/0.05
            scorepart_x = np.squeeze(scorepart_x)
            #grad_f = np.real(At(grad_phi(Ax, y, b)))[:N] + reg1 * diff2d_adj(grad_huber_v(Tx, reg2), sn, sn)
            
            
            # Adk = A(holocat(grad_f, np.zeros_like(grad_f)))
            # D1 = np.sqrt(fisher(Ax, b))
            # mu = - (norm(grad_f)**2)/ (norm(np.multiply(Adk, D1))**2) * (sigmas[iter]/0.05)**2
            #mu = - (norm(grad_f)**2) / (norm(np.multiply(Adk, D1))**2 + reg1 * (norm(np.multiply(Tdk, D2))**2))
            mu = -0.08*(sigmas[iter]**2) # set to -0.49 for purple dataset
            # mu = -(sigmas[iter]**2)/4
            #################### FGM updates #####################
            
            w = x + tn_old/tn * (z - x) + (tn_old-1)/tn * (x - x_old) # eqn.(10)
            w = np.clip(w, 0, 1)
            Aw = A(holocat(w, ref))
            
            # compute score(w)
            netinput_w = torch.from_numpy(np.reshape(w, (1,1,lsize,lsize))).float().to(next(model.parameters()).device)
            scorepart_w = -model.forward(netinput_w, network_sigma).cpu().detach().numpy().reshape((lsize, lsize))/sigmas[iter]
            scorepart_w = scorepart_w.reshape(-1)
            #scorepart = -model.forward(torch.from_numpy(x.reshape((lsize, lsize)))).cpu().detach().numpy()/0.05
            scorepart_w = np.squeeze(scorepart_w)
            
            grad_fw = np.real(At(grad_fun(Aw, y, b)))[:N] + 1*scorepart_w
            z = w + (mu * grad_fw) # eqn.(11)
            # do multiple realizations of LD and compute the average and std
            grad_fx = np.real(At(grad_fun(Ax, y, b)))[:N] + 1*scorepart_x
            vv = x + (mu * grad_fx) # eqn.(12) # for langevin dynamics: + np.sqrt(-2*mu)*np.random.randn(len(x))
            
            tn_old = np.copy(tn)
            tn = 1/2 * (1 + np.sqrt(1 + 4*(tn**2))) # eqn.(13)
            
            x_old = np.copy(x)
            # eqn.(14)
            x = gamma * z + (1-gamma) * vv
            # if np.sum(cost_fun(A(holocat(z, ref)), y, b)) <= np.sum(cost_fun(A(holocat(vv, ref)), y, b)):
            #     x = z
            # else:
            #     x = vv
            x = np.clip(x, 0, 1)
            Ax = A(holocat(x, ref))

        out.append(nrmse(x, xtrue))
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || nrmse (out, xtrue): {out[-1]:.4f}')
    return x, out
