import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import h5py
import scipy.io as sio
from random import randint
from utils3 import *
import math

class scoreloader(data.Dataset):
    def __init__(self, projdir = '.', psize = 3, patchesPerImage = 2, sigma = 0.05, imPerPatch = 10,
                       mode = 'train', noisy_im = []):
        
        self.mode = mode
        self.psize = psize
        super(scoreloader, self).__init__()
        self.sigma = sigma

        mat_contents = sio.loadmat(projdir)
        y = (mat_contents['x']*2)-1

        self.lsize = len(y[0,:,0])
        numimages = len(y[0,0,:])
        self.numimages = len(y[0,0,:])

        totalbetas = 100
        betas = np.linspace(1e-4, 0.3, num=totalbetas)
        alphas = 1-betas
        alphabars = np.cumprod(alphas)
        usebetas = 100

        trainx = np.zeros((self.lsize**2+1, numimages * usebetas))
        trainy = np.zeros((self.lsize, self.lsize, numimages * usebetas))
        #sigma = 0.05
        k = 0
        for i in range(numimages):
            for j in range(usebetas):
                alphabar = alphabars[j]
                x0 = np.squeeze(y[:,:,i]).copy()
                eps = np.random.normal(scale = 1, size = (self.lsize, self.lsize)) 
                inside = math.sqrt(alphabar)*x0 + math.sqrt(1-alphabar)*eps
                trainx[:self.lsize**2,k] = inside.flatten()
                trainy[:,:,k] = eps
                trainx[-1,k] = j+1
                k = k+1
        self.input = trainx.T
        self.output = trainy.reshape(self.lsize**2, numimages*usebetas).T

        # dir = '/n/newberry/v/jashu/grayscale_denoise/denoising-diffusion-pytorch/denoising_diffusion_pytorch/wholePhantoms.mat'
        # mat_contents = sio.loadmat(dir)


    def getsize(self):
        return self.psize, self.lsize

    def __len__(self):
        return len(self.input[:,0])

    def __getitem__(self, item):
        Meta = {}
        # if self.mode == 'train':
        #     Meta['input'] = torch.from_numpy(self.input[:,:,item]).to(torch.float)
        #     Meta['output'] = torch.from_numpy(self.output[:,:,item]).to(torch.float)    
        # else:
        #     Meta['input'] = torch.from_numpy(self.input[:,:,item]).to(torch.float)
            #Meta['output'] = torch.from_numpy(self.input[item,:]).to(torch.float)
        if self.mode == 'train':
            Meta['input'] = torch.reshape(torch.from_numpy(self.input[item,:self.lsize**2]).to(torch.float), (1, self.lsize, self.lsize))
            Meta['time'] = torch.Tensor([self.input[item,-1]])
            Meta['output'] = torch.reshape(torch.from_numpy(self.output[item,:]).to(torch.float), (self.lsize, self.lsize))
        else:
            Meta['input'] = torch.from_numpy(self.input[item,:]).to(torch.float)
            #Meta['output'] = torch.from_numpy(self.input[item,:]).to(torch.float)
        return Meta

if __name__ == "__main__":
    path = '../data/downsampledPhantoms.mat'
    loader = scoreloader(projdir = path, sigma = 0.005)
    #print('length of loader: ', len(loader))
    #print('a sample from loader: ', loader[10])
    print(loader.input[0:5, :])
    mdic = {"input": loader.input, "output": loader.output}
    sio.savemat("in_out.mat", mdic)