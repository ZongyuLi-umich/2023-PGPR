import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import h5py
import scipy.io as sio
from random import randint
from utils import *

class scoreloader(data.Dataset):
    def __init__(self, projdir = '.', psize = 3, patchesPerImage = 2, sigma = 0.05, imPerPatch = 10,
                       mode = 'train', noisy_im = []):
        
        self.mode = mode
        self.psize = psize
        self.lsize = -1 #this should never be used, throw an error if it does
        super(scoreloader, self).__init__()
        self.sigma = sigma

        # if mode == 'test':
        #     #prepare a ~4000 by 9 array of all the patches and just return it
        #     self.input = noisy_im.reshape(-1, psize**2)
        #     return
        # #f = h5py.File(projdir)
        # #y = np.array(f['y'])
        # mat_contents = sio.loadmat(projdir)
        # self.y = mat_contents['x']
        
        # #print(self.y.shape) #this is 64 by 64 by 1000 as expected, now get the patches

        # self.numimages = len(self.y[0,0,:])
        # self.lsize = len(self.y[0,:,0])
        # patchArr = np.zeros((psize, psize, patchesPerImage * self.numimages))
        # a = 0
        # for i in range(self.numimages):
        #     myim = np.squeeze(self.y[:,:,i])
        #     for j in range(patchesPerImage):
        #         row = randint(0, self.lsize - psize)
        #         col = randint(0, self.lsize - psize)
        #         myPatch = myim[row:row+psize, col:col+psize].copy()
        #         patchArr[:,:,a] = myPatch.copy()
        #         a = a+1

        # #now add noise to each patch 
        # self.totalPts = patchesPerImage * self.numimages * imPerPatch
        # self.input = np.zeros((psize, psize, self.totalPts))
        # self.output = np.zeros((psize, psize, self.totalPts))
        # a = 0
        # for i in range(patchesPerImage * self.numimages):
        #     for j in range(imPerPatch):
        #         xhat = np.squeeze(patchArr[:,:,i]) + np.random.normal(scale = sigma, size = (psize, psize))
        #         self.input[:,:,a] = xhat.copy()
        #         self.output[:,:,a] = (np.squeeze(patchArr[:,:,i]) - xhat)/sigma**2
        #         a = a+1
        # self.output = self.output * sigma
        
        # # print(self.input[:,:,1500])
        # # print(self.output[:,:,1500])
        # self.input = self.input.reshape(psize**2, self.totalPts)
        # self.input = self.input.T
        # self.output = self.output.reshape(psize**2, self.totalPts)
        # self.output = self.output.T
        # # print(self.output.shape)
        # # print(self.input.shape) something by 9

        mat_contents = sio.loadmat(projdir)
        #mat_contents = h5py.File(projdir)
        y = mat_contents['x']
        #y = y.value
        #y = y.T
        self.lsize = len(y[0,:,0])
        numimages = len(y[0,0,:])
        self.numimages = len(y[0,0,:])
        totalsigmas = 20
        trainx = np.zeros((self.lsize**2+1, numimages * totalsigmas))
        trainy = np.zeros((self.lsize, self.lsize, numimages * totalsigmas))
        #sigma = 0.05
        sigmas = np.geomspace(0.005, 0.1, num=totalsigmas)
        k = 0
        for i in range(numimages):
            for j in range(len(sigmas)):
                sigma = sigmas[j]
                clean = np.squeeze(y[:,:,i]).copy()
                xhat = clean + np.random.normal(scale = sigma, size = (self.lsize, self.lsize)) 
                trainx[:self.lsize**2,k] = xhat.flatten()
                trainy[:,:,k] = (clean - xhat)/sigma
                trainx[-1,k] = sigma
                k = k+1
        self.input = trainx.T
        self.output = trainy.reshape(self.lsize**2, numimages*totalsigmas).T

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