# pnp_pgred_noise2self.py
# plug and play RED method for Poisson+Gaussian Phase Retrieval
from ast import AugLoad
import copy
from utils2 import * 
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from eval_metric import *
import torch
from noise2self import Masker
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam


def pnp_pgred_noise2self(A, At, y, b, x0, ref, sigma, delta, niter, 
                         xtrue, model, mu = None, scale = 1, rho=1, verbose=True):
    
    # xiaojian
    # x0 = denoise(x0, model, scale, sn=128)
    masker = Masker(width = 4, mode='interpolate')
    N = len(x0)
    sn = np.sqrt(N).astype(int)
    out = []
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    Ax = A(holocat(x, ref))
    _, grad_phi, fisher = get_grad(sigma, delta)
    print('Deep copy the pre-trained model and finetune it!')
    model_copy = copy.deepcopy(model)
    ########## pretrain on x0 ################
    model_tuned = pretrain(noisy=x, model=model_copy, masker=masker, 
                           niter=200, sn=sn)
    print('********finetune finished!*********')

    lastnrmse = 1
    for iter in range(niter):
        grad_x = np.real(At(grad_phi(Ax, y, b)))[:N]
        
        if mu is None:
            Adk = A(holocat(grad_x, np.zeros_like(grad_x)))
            D1 = np.sqrt(fisher(Ax, b))
            mu = - (norm(grad_x)**2)/ (norm(np.multiply(Adk, D1))**2) 
                
        Dx = denoise(x, model_tuned, sn=sn)
        x = x + mu *  (grad_x + rho * (x - Dx))
        x = np.clip(x, 0, 1)
                
        Ax = A(holocat(x, ref))
        out.append(nrmse(x, xtrue))
        
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || scale: {1.0} || step: {mu:.2e} || nrmse (out, xtrue): {out[-1]:.4f}')

    return x, out


def pretrain(noisy, model, masker, niter, sn=128):
    noisy = np.clip(noisy, 0, 1)
    noisy = torch.from_numpy(jreshape(noisy, sn, sn))[None, None, ...].to(torch.float32).cuda()
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = MSELoss()
    for i in range(niter):
        net_input, mask = masker.mask(noisy, i % (masker.n_masks - 1))
        net_output = model(net_input)
        loss = loss_function(net_output*mask, noisy*mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model
        
    
def denoise(x_tmp, model, sn=128):
    with torch.no_grad():
        x_tmp = np.clip(x_tmp, 0, 1)
        x_tmp = torch.from_numpy(jreshape(x_tmp, sn, sn))[None, None, ...].to(torch.float32).cuda()
        x_tmp = model(x_tmp)
        x_tmp = vec(x_tmp.cpu().numpy())
        x_tmp = np.clip(x_tmp, 0, 1)
        return x_tmp
    
