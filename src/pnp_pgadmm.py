# pnp_pgadmm.py
# plug and play ADMM method for Poisson+Gaussian Phase Retrieval
from utils2 import * 
import numpy as np
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


def pnp_pgadmm(A, At, y, b, x0, ref, sigma, delta, niter, rho, prox, uiter, xtrue):
    N = len(x0)
    out = []
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    u = np.copy(x)
    Au = A(holocat(u, ref))
    eta = np.zeros(N)
    phi, grad_phi, fisher = get_grad(sigma, delta)
    for _ in tqdm(range(niter)):
        # update u
        def cost_fun_u(uu): return np.sum(phi(abs2(A(holocat(uu, ref)))+b, y, sigma, delta)) \
                            + rho/2 * (norm(x - uu + eta)**2)
        for _ in range(uiter):
            grad_u = 2 * np.real(At(np.multiply(grad_phi(abs2(Au)+b, y, sigma, delta), Au)))[:N] \
                        + rho * (u - x - eta) 
                    
            mu_u_hat = 0.001 # assume u is non-negative
        
            u = np.maximum(0, u - mu_u_hat * grad_u)
            # update Au
            Au = A(holocat(u, ref))
        # update x
        ##########################################
        # Implement plug and play
        x = prox(x, u, eta, rho) 
        ##########################################
        # update eta
        eta = eta + x - u
        out.append(nrmse(x, xtrue))
        print('nrmse: ', out[-1])
    return x, out