import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import scipy.io as sio
from utils import *
from test import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from model2 import *
from test import scoretester


def normalize(arr, getone = False):
    #get an array between 0 and 255
    arr = np.minimum(arr, np.ones_like(arr))
    arr = np.maximum(arr, np.zeros_like(arr))
    if getone:
        return arr
    arr = (arr * 255.99).astype(np.uint8)
    return arr

def getDivision(lsize, psize, overlap):
    overlaparr = np.zeros((lsize, lsize))
    for i in range(0, lsize-psize+1, overlap):
        for j in range(0, lsize-psize+1, overlap):
            bruh = np.zeros((lsize, lsize))
            bruh[i:i+psize, j:j+psize] = 1
            overlaparr = overlaparr + bruh
    
    for i in range(lsize):
        for j in range(lsize):
            if overlaparr[i,j] == 0:
                overlaparr[i,j] == 1
    return overlaparr


def score(x, overlaparr, overlap, psize, model):
    grad = np.zeros_like(x)
    lsize = len(x[:,0])

    tester = scoretester(model, is_denoise = True, im = x)
    first = tester.test()
    allscore = first.reshape(-1, psize**2) #score function for every patch
    #print(allscore.shape)
    a = 0
    for i in range(0, lsize-psize+1, overlap):
        for j in range(0, lsize-psize+1, overlap):
            Gc = getPatch(x, i, j, psize)
            sV = allscore[a,:]
            grad = grad + GcT(sV, i, j, psize, lsize)
            a = a+1
    #print(np.min(grad), np.max(grad))
    #return -np.divide(grad,overlaparr)
    return -grad

def denoise2(image, clean, args = '', sigma = 0.05, tau = 1, checkpoint_dir = '', show = True, iters=100, network_sigma=0.05, alpha = 0.0001):
    parser = config_parser()
    args = parser.parse_args()
    lsize = len(image[:,0])
    images = np.zeros((iters, lsize, lsize))
    overlaparr = getDivision(lsize, args.patchsize, args.overlap)
    
    xk = image.copy() #this is a noisy image
    lastpsnr = 0

    iters = 10
    T = 30
    sigmas = np.geomspace(0.06, 0.002, iters)
    # st = score(xk, overlaparr, args.overlap, args.patchsize, checkpoint_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=lsize)
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    model.to(device)
    curpsnr = psnr(normalize(xk), normalize(clean))
    if show:
        print(curpsnr)
    for i in range(1, iters):
        netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
        network_sigma = torch.from_numpy(np.array([sigmas[i]])).float().cuda()
        #print(network_sigma)
        scorepart = model.forward(netinput, network_sigma).cpu().detach().numpy().reshape((lsize, lsize))
        alpha = sigmas[i]**2/10
        lam = alpha * 50
        xk = xk + (sigmas[i-1]**2 - sigmas[i]**2) * scorepart/sigmas[i]
        xk = xk + lam * (image - xk)

        for j in range(T):
            # netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
            # network_sigma = torch.from_numpy(np.array([sigmas[i]])).float().cuda()
            # #print(network_sigma)
            # scorepart = model.forward(netinput, network_sigma).cpu().detach().numpy().reshape((lsize, lsize))
            z = np.random.randn(lsize, lsize)
            xk = xk + alpha * scorepart/sigmas[i] + 0*math.sqrt(2*alpha)*z
            xk = xk + lam * (image - xk)

        curpsnr = psnr(normalize(xk), normalize(clean))
        if show:
            print(curpsnr)
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,1),plt.imshow(normalize(image),cmap='gray'),plt.axis('off'),plt.title('noisy')
            plt.subplot(1,3,2),plt.imshow(normalize(xk),cmap='gray'),plt.axis('off'),plt.title('denoised')
            plt.subplot(1,3,3),plt.imshow(normalize(clean),cmap='gray'),plt.axis('off'),plt.title('clean')
            plt.show()
            plt.savefig('../results/' + str(i) + '.png')
            plt.close('all')
        if abs(curpsnr - lastpsnr) < 0.001:
        #if i == iters-1:
            #try a final scaling trick 
            xk = findScale(xk, clean)
            print(psnr(normalize(xk), normalize(clean)))
            print('the best snr is ', snr(xk, clean))
            break
        lastpsnr = curpsnr
    return xk


def denoise(image, clean, args = '', sigma = 0.05, tau = 1, checkpoint_dir = '', show = True, iters=100, network_sigma=0.05, alpha = 0.0001):
    parser = config_parser()
    args = parser.parse_args()
    lsize = len(image[:,1])
    images = np.zeros((iters, lsize, lsize))
    overlaparr = getDivision(lsize, args.patchsize, args.overlap)
    
    xk = image.copy() #this is a noisy image
    lastpsnr = 0

    iters = 10
    T = 10
    sigmas = np.geomspace(0.05, 0.005, iters)
    eps = 9e-6
    # st = score(xk, overlaparr, args.overlap, args.patchsize, checkpoint_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=lsize)
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    model.to(device)
    curpsnr = psnr(normalize(xk), normalize(clean))
    if show:
        print(curpsnr)
    for i in range(iters):
        images[i,:,:] = xk
        #gd = (image - xk)/sigma**2 - score(xk, overlaparr, args.overlap, args.patchsize, model)/sigma #old score function
        netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
        network_sigma = torch.from_numpy(np.array([sigmas[i]])).float().cuda()
        #print(network_sigma)
        scorepart = -model.forward(netinput, network_sigma).cpu().detach().numpy().reshape((lsize, lsize))
        alpha = eps * (sigmas[i]/sigmas[-1])**2
        #print(alpha)
        for t in range(T):
            z = np.random.randn(lsize, lsize)
            c = 1
            if t == 0:
                c = 1
            else: 
                c = 0
            xk = xk + alpha/2*(c*(image-xk)/sigma**2 - scorepart/sigmas[i]) + 1*math.sqrt(alpha)*z
        #xk = findScale(xk, clean)
        #xk = normalize(xk, getone = True)

        curpsnr = psnr(normalize(xk), normalize(clean))
        if show:
            print(curpsnr)
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,1),plt.imshow(normalize(image),cmap='gray'),plt.axis('off'),plt.title('noisy')
            plt.subplot(1,3,2),plt.imshow(normalize(xk),cmap='gray'),plt.axis('off'),plt.title('denoised')
            plt.subplot(1,3,3),plt.imshow(normalize(clean),cmap='gray'),plt.axis('off'),plt.title('clean')
            plt.show()
            plt.savefig('/n/newberry/v/jashu/scoreMatchingFull/results/' + str(i) + '.png')
            plt.close('all')
        if abs(curpsnr - lastpsnr) < 0.001 or lastpsnr > curpsnr:
        #if i == iters-1:
            #try a final scaling trick 
            xk = findScale(xk, clean)
            print(psnr(normalize(xk), normalize(clean)))
            print('the best snr is ', snr(xk, clean))
            break
        lastpsnr = curpsnr
    return xk


from scipy import optimize
def optimizeTau(x, algoHandle, taurange, maxfun=20):
    evaluateSNR = lambda x, xhat: 20 * np.log10(np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))
    fun = lambda tau: -evaluateSNR(x, algoHandle(tau))
    tau = optimize.fminbound(fun, taurange[0], taurange[1], xtol=1e-3, maxfun=maxfun, disp=3)
    return tau


if __name__ == '__main__':

    init_env(seed_value=43)

    #mat_contents = sio.loadmat('../data/downsampledPhantoms.mat')
    # mat_contents = sio.loadmat('../data/downsampledPhantomsTest.mat')
    # mat_contents = sio.loadmat('../data/flowerclean.mat')
    #mat_contents = sio.loadmat('../data/coilimages.mat')
    mat_contents = sio.loadmat('/home/lizongyu/PycharmProjects/2023-PGPR/data/sem_small/sem_small_test.mat')

    image = mat_contents['x']
    clean = np.squeeze(image[:,:,50])
    sigma = 0.2
    lsize = len(image[:,0])
    noisy = clean + np.random.normal(scale = sigma, size = (lsize, lsize))

    # checkpoint_dir = "../logs/2023-02-22-11-44-33/models/best_checkpoint.pytorch" #flower network
    # checkpoint_dir = "../logs/2023-02-22-13-49-43/models/best_checkpoint.pytorch" #phantom network
    # checkpoint_dir = "../logs/2023-02-24-14-37-36/models/last_checkpoint.pytorch" #flower network 2 0.05
    # checkpoint_dir2 = "../logs/2023-02-24-14-24-25/models/best_checkpoint.pytorch" #flower network 2 0.01
    #checkpoint_dir = "../logs/2023-03-07-14-20-10/models/last_checkpoint.pytorch" #coil network
    checkpoint_dir = "../logs/cem500k-small/models/best_checkpoint.pytorch" #purple network
    # checkpoint_dir = "../logs/2023-03-13-23-30-11/models/best_checkpoint.pytorch" #ncsn network purple
    #checkpoint_dir = "../logs/2023-03-14-14-25-27/models/best_checkpoint.pytorch" #ncsn network nature
    #checkpoint_dir = "../logs/2023-03-14-22-53-39/models/best_checkpoint.pytorch" #ncsn network coil
    out = denoise2(noisy, clean, sigma = sigma, tau = 1, checkpoint_dir = checkpoint_dir, network_sigma=0.05, alpha = 0.0002)
    #out = denoise(out, clean, sigma = sigma, tau = 1, checkpoint_dir = checkpoint_dir2, network_sigma=0.01, alpha = 0.00005)

    # taurange = [100, 1000]
    # algoHandle = lambda tau : denoise(noisy, clean, sigma = sigma, tau=tau, checkpoint_dir = checkpoint_dir)
    # tau = optimizeTau(clean, algoHandle, taurange, maxfun=20)
    # print(f'optimized tau = {tau}')



