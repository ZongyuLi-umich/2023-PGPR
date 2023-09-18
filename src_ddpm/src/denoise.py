import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import scipy.io as sio
from utils3 import *
from test import *
from skimage.metrics import peak_signal_noise_ratio as psnr
from model2 import *

def normalize(arr, getone = False):
    #get an array between 0 and 255
    arr = np.minimum(arr, np.ones_like(arr))
    arr = np.maximum(arr, np.zeros_like(arr))
    if getone:
        return arr
    arr = (arr * 255.99).astype(np.uint8)
    return arr

def timetravel(image, clean, sigma = 0.05, checkpoint_dir = '', show = True, iters=100):
    parser = config_parser()
    args = parser.parse_args()
    lsize = len(image[:,0])
    images = np.zeros((iters, lsize, lsize))

    xk = image.copy() #this is a noisy image
    lastpsnr = 0
    bestpsnr = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=lsize)
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    model.to(device)
    curpsnr = psnr(normalize(xk), normalize(clean))

    totalbetas = 100
    betas = np.linspace(1e-4, 0.3, num=totalbetas)
    alphas = 1-betas
    alphabars = np.cumprod(alphas)

    xk = np.random.randn(lsize, lsize)/2+1
    for t in range(10, 0, -1):
        if t >= 5:
            rt = 5
        elif t >= 3:
            rt = 10
        elif t == 2:
            rt = 10
        elif t == 1: 
            rt = 10
        else: 
            rt = 10
        for i in range(rt, 0, -1):
            if i > 0:
                eps1 = np.random.randn(lsize, lsize)
            else:
                eps1 = np.zeros((lsize, lsize))
            xk = xk*2-1
            netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
            network_t = torch.from_numpy(np.array([t])).float().cuda()
            #print(network_t)
            epspart = model.forward(netinput, network_t).cpu().detach().numpy().reshape((lsize, lsize))

            oldx = xk.copy()
            xk = (1+betas[t-1]/2) * xk + betas[t-1]*epspart + math.sqrt(betas[t-1]) * eps1*0
            x0givent = 1/math.sqrt(alphabars[t-1]) * (oldx + (1-alphabars[t-1])*epspart)

            xk = (xk+1)/2
            lam = 0.05#0.05*((1-alphas[t-1])/math.sqrt(1-alphabars[t-1]))
            xk = xk + lam*(image-x0givent) #data fidelity

            if i > 0:
                eps2 = np.random.randn(lsize, lsize)
                xk = xk*2-1
                xk = math.sqrt(1-betas[t-1])*xk + math.sqrt(betas[t-1])*eps2*0
                xk = (xk+1)/2

        #xk = normalize(xk, getone=True)
        curpsnr = psnr(normalize(xk), normalize(clean))
        if show:
            print(curpsnr)
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,1),plt.imshow(normalize(image),cmap='gray'),plt.axis('off'),plt.title('noisy')
            plt.subplot(1,3,2),plt.imshow(normalize(xk),cmap='gray'),plt.axis('off'),plt.title('denoised')
            plt.subplot(1,3,3),plt.imshow(normalize(clean),cmap='gray'),plt.axis('off'),plt.title('clean')
            plt.show()
            plt.savefig('../results/' + str(t) + '.png')
            plt.close('all')
        # if abs(curpsnr - lastpsnr) < 0.001 or lastpsnr > curpsnr:
        # #if i == iters-1:
        #     #try a final scaling trick 
        #     xk = findScale(xk, clean)
        #     print(psnr(normalize(xk), normalize(clean)))
        #     print('the best snr is ', snr(xk, clean))
        #     break
        lastpsnr = curpsnr
    return xk

def sample(lsize, checkpoint_dir = '', show = True):
    parser = config_parser()
    args = parser.parse_args()
    xk = np.random.normal(scale = 1, size = (lsize, lsize)) #this is a noisy image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=lsize)
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    model.to(device)

    totalbetas = 100
    betas = np.linspace(1e-4, 0.3, num=totalbetas)
    alphas = 1-betas
    alphabars = np.cumprod(alphas)

    for t in range(totalbetas, 0, -1):
        netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
        network_t = torch.from_numpy(np.array([t])).float().cuda()
        #print(network_t)
        epspart = model.forward(netinput, network_t).cpu().detach().numpy().reshape((lsize, lsize))

        if t > 1:
            z = np.random.randn(lsize, lsize)
        else:
            z = np.zeros((lsize, lsize))
        sigma_t = math.sqrt(betas[t-1])
        xk = (xk - (1-alphas[t-1])/math.sqrt(1-alphabars[t-1]) * epspart)/math.sqrt(alphas[t-1]) + sigma_t * z
        #xk = normalize(xk/2, getone=True)*2
        if show and t%10 == 0:
            print(t)
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,2),plt.imshow(normalize(xk),cmap='gray'),plt.axis('off'),plt.title('denoised')
            plt.show()
            plt.savefig('../results/' + str(t) + '.png')
            plt.close('all')
        # if abs(curpsnr - lastpsnr) < 0.001 or lastpsnr > curpsnr:
        # #if i == iters-1:
        #     #try a final scaling trick 
        #     xk = findScale(xk, clean)
        #     print(psnr(normalize(xk), normalize(clean)))
        #     print('the best snr is ', snr(xk, clean))
        #     break
    return xk

def denoise2(image, clean, checkpoint_dir = '', show = True):
    parser = config_parser()
    args = parser.parse_args()
    lsize = len(image[:,0])

    xk = image.copy() #this is a noisy image
    lastpsnr = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=lsize)
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    model.to(device)
    curpsnr = psnr(normalize(xk), normalize(clean))

    totalbetas = 1000
    betas = np.linspace(5e-5, 0.01, num=totalbetas)
    alphas = 1-betas
    alphabars = np.cumprod(alphas)

    t = 199
    netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
    network_t = torch.from_numpy(np.array([t])).float().cuda()
    #print(network_t)
    epspart = model.forward(netinput, network_t).cpu().detach().numpy().reshape((lsize, lsize))
    xk = 1/math.sqrt(alphas[t-1]) * (xk - betas[t-1]/math.sqrt(1-alphabars[t-1]) * epspart)
    xk = normalize(xk, getone=True)
    curpsnr = psnr(normalize(xk), normalize(clean))
    if show:
        print(curpsnr)
        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1),plt.imshow(normalize(image),cmap='gray'),plt.axis('off'),plt.title('noisy')
        plt.subplot(1,3,2),plt.imshow(normalize(xk),cmap='gray'),plt.axis('off'),plt.title('denoised')
        plt.subplot(1,3,3),plt.imshow(normalize(clean),cmap='gray'),plt.axis('off'),plt.title('clean')
        plt.show()
        plt.savefig('../results/dewei.png')
        plt.close('all')

def denoise(image, clean, sigma = 0.05, checkpoint_dir = '', show = True, iters=100):
    parser = config_parser()
    args = parser.parse_args()
    lsize = len(image[:,0])
    images = np.zeros((iters, lsize, lsize))

    xk = image.copy() #this is a noisy image
    lastpsnr = 0
    bestpsnr = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(dim=lsize)
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    model.to(device)
    curpsnr = psnr(normalize(xk), normalize(clean))

    totalbetas = 100
    betas = np.linspace(1e-4, 0.3, num=totalbetas)
    alphas = 1-betas
    alphabars = np.cumprod(alphas)

    for t in range(12, 0, -1):
        xk = xk*2-1
        netinput = torch.from_numpy(np.reshape(xk, (1,1,lsize,lsize))).float().cuda()
        network_t = torch.from_numpy(np.array([t])).float().cuda()
        #print(network_t)
        epspart = model.forward(netinput, network_t).cpu().detach().numpy().reshape((lsize, lsize))

        if t > 1:
            z = np.random.randn(lsize, lsize)
        else:
            z = np.zeros((lsize, lsize))
        sigma_t = math.sqrt(betas[t-1])*0 #one possible choice
        # if t >= 2:
        #     sigma_t = 0*math.sqrt((1-alphabars[t-2])/(1-alphabars[t-1])*betas[t-1])
        # else: 
        #     sigma_t = 0

        xk = (xk - (1-alphas[t-1])/math.sqrt(1-alphabars[t-1]) * epspart)/math.sqrt(alphas[t-1]) + sigma_t * z

        xk = (xk+1)/2
        lam = 0.01#0.05*((1-alphas[t-1])/math.sqrt(1-alphabars[t-1]))
        xk = xk + lam*(image-xk) #data fidelity

        xk = normalize(xk, getone=True)
        curpsnr = psnr(normalize(xk), normalize(clean))
        if show:
            print(curpsnr)
            plt.figure(figsize=(18,6))
            plt.subplot(1,3,1),plt.imshow(normalize(image),cmap='gray'),plt.axis('off'),plt.title('noisy')
            plt.subplot(1,3,2),plt.imshow(normalize(xk),cmap='gray'),plt.axis('off'),plt.title('denoised')
            plt.subplot(1,3,3),plt.imshow(normalize(clean),cmap='gray'),plt.axis('off'),plt.title('clean')
            plt.show()
            plt.savefig('../results/' + str(t) + '.png')
            plt.close('all')
        # if abs(curpsnr - lastpsnr) < 0.001 or lastpsnr > curpsnr:
        # #if i == iters-1:
        #     #try a final scaling trick 
        #     xk = findScale(xk, clean)
        #     print(psnr(normalize(xk), normalize(clean)))
        #     print('the best snr is ', snr(xk, clean))
        #     break
        lastpsnr = curpsnr
    return xk

if __name__ == '__main__':

    init_env(seed_value=43)

    #mat_contents = sio.loadmat('../data/downsampledPhantoms.mat')
    # mat_contents = sio.loadmat('../data/downsampledPhantomsTest.mat')
    # mat_contents = sio.loadmat('../data/flowerclean.mat')
    #mat_contents = sio.loadmat('../data/coilimages.mat')
    mat_contents = sio.loadmat('/home/lizongyu/PycharmProjects/2023-PGPR/data/celebA_small/celebA_small_test.mat')

    image = mat_contents['x']
    clean = np.squeeze(image[:,:,10])
    sigma = 0.2
    lsize = len(image[:,0])
    noisy = clean + np.random.normal(scale = sigma, size = (lsize, lsize))

    #checkpoint_dir = "../logs/2023-04-04-13-44-34/models/best_checkpoint.pytorch" #purple 100
    #checkpoint_dir = "../logs/2023-04-04-17-45-08/models/best_checkpoint.pytorch" #purple 1000
    #checkpoint_dir = "../logs/2023-04-05-10-42-28/models/best_checkpoint.pytorch" #purple but -1 1
    checkpoint_dir = "../logs/celebA_small/models/best_checkpoint.pytorch" #purple 100
    #sample(128, checkpoint_dir = checkpoint_dir)
    #out = denoise2(noisy, clean, checkpoint_dir = checkpoint_dir)
    out = timetravel(noisy, clean, sigma = sigma, checkpoint_dir = checkpoint_dir)
    