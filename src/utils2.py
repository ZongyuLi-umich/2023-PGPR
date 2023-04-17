# utils.py
# useful functions for PGADMM
import numpy as np
from scipy.special import lambertw
from numpy.linalg import norm
import os

def check_and_mkdir(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)
                    
def abs2(x):
    return np.abs(x)**2

def cal_nmax(a, b, sigma, delta):
    if a/(sigma**2) > 0:
        nstar = sigma * np.real(lambertw(a/(sigma**2) * np.exp(b/(sigma**2))))
    else:
        nstar = sigma
    # if a/(sigma**2) >1 and b > 0:
    #     nstar = b/sigma * np.log(a/(sigma**2)) - sigma * np.log(b/(sigma**2)*np.log(a/(sigma**2)))
    # else:
    #     nstar = sigma
    if np.isnan(nstar):
        nstar = sigma
    nmax = np.ceil(nstar + delta * sigma).astype(int)
    nmin = np.floor(nstar - delta * sigma).astype(int)
    return nmax, nmin

def compute_grad_s(a, b, sigma, delta):
    nmax, nmin = cal_nmax(a, b, sigma, delta)
    # print('nmax: ', nmax)
    # print('a: ', a)
    # print('b: ', b)
    # scalefact = np.maximum(a, b)
    # s = np.exp(- b**2 / (2*(sigma**2)) - scalefact) + 1e-8
    s = 0
    for n in range(0, nmax+1):
        log_s = n * np.log(a) - np.sum(np.log(np.arange(1, n+1))) \
                    - (b-n)**2 / (2*(sigma**2))
        s += np.exp(log_s)
    # if nmax < 50:
    #     for n in range(1, nmax+1):
    #         log_s = n * np.log(a) - np.sum(np.log(np.arange(1, n+1))) \
    #                 - (b-n)**2 / (2*(sigma**2))
    #         s += np.exp(log_s - scalefact)
    # else:
    #     for n in range(nmin, nmax+1):
    #         log_s = n * np.log(a) - np.sum(np.log(np.arange(1, n+1))) \
    #                 - (b-n)**2 / (2*(sigma**2))
    #         s += np.exp(log_s - scalefact)
    return s

def grad_phi1(u, y, sigma, delta):
    return 1 - compute_grad_s(u, y-1, sigma, delta) / compute_grad_s(u, y, sigma, delta)

def phi1(u, y, sigma, delta):
    nmax = cal_nmax(u, y, sigma, delta)
    s = 0
    for n in range(0, nmax+1):
        log_s = n * np.log(u) - u - np.sum(np.log(np.arange(1, n+1))) \
                - (y-n)**2 / (2 * sigma**2) - np.log(np.sqrt(2*np.pi*(sigma**2)))
        s += np.exp(log_s)
    return -np.log(s)

def huber(t, a):
    if np.abs(t) < a:
        return 0.5 * (np.abs(t)**2)
    else:
        return a * np.abs(t) - 0.5 * (a**2)

def grad_huber(t, a):
    if np.abs(t) < a:
        return t
    else:
        return a * np.sign(t)

def curv_huber(t, a):
    if np.abs(t) > a:
        return a / np.abs(t)
    else:
        return 1
    
def vec(x):
    return x.T.reshape(-1)    

def jreshape(x, M, N):
    return x.reshape(N, M).T 

def diff2d_forw(x, M, N):
    x = jreshape(x, M, N)
    d1 = vec(np.diff(x, axis = 0))
    d2 = vec(np.diff(x, axis = 1))
    return np.concatenate((d1, d2))

def diff2d_adj(x, M, N):
    d1 = jreshape(x[0:(M-1)*N], M-1, N)
    d2 = jreshape(x[(M-1)*N:], M, N-1)
    z1 = vec(np.concatenate((jreshape(-d1[0,:],1,N), -np.diff(d1, axis=0), jreshape(d1[-1,:],1,N)), axis=0))
    z2 = np.concatenate((-d2[:,0], vec(-np.diff(d2, axis=1)), d2[:,-1]))
    return z1 + z2    

def power_iter(A, x, niter):
    for iter in range(niter):
        x = A(x)
        x /= np.linalg.norm(x)
    return x

def backtracking(mu0, x, costfun, grad_x, 
                 lowerbound = -np.Inf, 
                 mustep = 0.01, mushrink = 2):
    x_old = np.copy(x)
    mu = np.copy(mu0)
    cost_x_old = costfun(x_old)
    x_new = np.maximum(lowerbound, x - mu * grad_x)
    while costfun(x_new) > cost_x_old - mu * mustep * (norm(grad_x)**2):
        mu = mu / mushrink
        x_new = np.maximum(lowerbound, x_old - mu * grad_x)
    return mu

def pad_fft(x, M, N, K, L):
    # pad from M*N to K*L
    x_reshaped = jreshape(x, M, N)
    hcatx = np.hstack((x_reshaped, np.zeros((M, L-N))))    
    vcatx = np.vstack((hcatx, np.zeros((K-M, L))))
    return vec(np.fft.fft2(vcatx))   

def unpad_ifft(y, M, N, K, L):
    # unpad from K*L to M*N
    y_reshaped = jreshape(y, K, L)
    y_ifft = np.fft.ifft2(y_reshaped)
    y_unpad = y_ifft[:M, :N]
    return (K*L)*vec(y_unpad)

def holocat(x, ref):
    N = len(x)
    sn = np.sqrt(N).astype(int)
    x_reshaped = jreshape(x, sn, sn)
    ref_reshaped = jreshape(ref, sn, sn)
    concated = np.hstack((x_reshaped, np.zeros_like(x_reshaped), ref_reshaped)) # holographic separation condition
    return vec(concated)

def get_grad(sigma, delta):
    def phi(v, yi, bi): return (abs2(v) + bi) - yi * np.log(abs2(v) + bi)
    def grad_phi(v, yi, bi): 
        if yi < np.minimum(100, 100/(sigma**2)):
            u = abs2(v) + bi
            return 2 * v * grad_phi1(u, yi, sigma, delta)
        else:
            return 2 * v * (1 - yi / (abs2(v) + bi))
    def fisher(vi, bi): return 4 * abs2(vi) / (abs2(vi) + bi)
    return np.vectorize(phi), np.vectorize(grad_phi), np.vectorize(fisher)

if __name__ == "__main__":
    # test fft
    N = 64
    M = 3 * N
    L = 128
    K = 3 * L
    x = vec(np.random.randn(M, N))
    Ax = pad_fft(x, M, N, K, L)
    y = vec(np.random.randn(K, L))
    Aty = unpad_ifft(y, M, N, K, L)
    print(np.vdot(Ax, y))
    print(np.vdot(x, Aty))
    # x = vec(np.random.randn(N, N))
    # ref = vec(np.random.randn(N, N))
    # Ax = holocat(x, ref)
    # y = vec(np.random.randn(M, N))
    
    # ref = vec(np.random.randn(N, N))
    # ref = vec(np.zeros_like(x))
    # Ax = fft_forw(x, ref, M, N, K, L)
    
    # y = vec(np.random.randn(K, L))
    # Aty = fft_adj(y, ref, M, N, K, L)
    # print(np.vdot(Ax, y))
    # print(np.vdot(x, Aty))
    
    # vec_x = vec(np.random.randn(N, N))
    # x_cated = holocat(vec_x, np.zeros_like(vec_x))
    # print(np.linalg.norm(vec_x - x_cated[:len(vec_x)]))
    # u = 1
    # y = 0.8
    # sigma = 1
    # delta = 0.1
    # # check with Julia implementation
    # print('grad phi:', grad_phi1(u, y, sigma, delta))
    # print('phi:', phi1(u, y, sigma, delta))
    
    # M = 5
    # N = 4
    # x = np.random.randn(M*N)
    # y = np.random.randn((M-1)*N + M*(N-1))
    # x_forw = diff2d_forw(x, M, N)
    # y_adj = diff2d_adj(y, M, N)
    # print(np.dot(y, x_forw))
    # print(np.dot(x, y_adj))
