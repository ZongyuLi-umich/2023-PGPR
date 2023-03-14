# dataloader.py
import tifffile as tiff
import cv2
import numpy as np
import os
from PIL import Image
from utils import *
import scipy.io as sio

class PrDataset:
    def __init__(self, path, N, scalefact, sigma) -> None:
        self.path = path
        self.data = {'xtrue': [], 'ynoisy': [], 'x0': []}
        self.N = N
        self.M = 3 * N
        self.K = 2 * self.M
        self.L = 2 * N
        self.scalefact = scalefact
        self.img_format = ['png', 'jpg', 'jpeg', 'tiff']
        self.mat_format = 'test.mat'
        self.load_data()
        print('# of img: ', len(self.data['xtrue']))
        self.ref = np.random.randint(2, size=(N**2,))
        self.b = 0.1 * np.ones(self.K*self.L)
        self.sigma = sigma
        self.A, self.At = self.gen_sys()
        self.gen_data(self.A, self.At)
    def load_data(self):
        for (dirpath, dirnames, filenames) in os.walk(self.path):
            for filename in filenames:
                if filename.split('.')[-1] in self.img_format: 
                    print('load data from: ', os.path.join(dirpath, filename))
                    if filename.endswith('.tiff'):
                        img = tiff.imread(os.path.join(dirpath, filename))
                    else:
                        img = np.array(Image.open(os.path.join(dirpath, filename)))
                    res_img = cv2.resize(img, dsize=(self.N, self.N))
                    xtrue = np.array(res_img) / np.max(res_img)
                    print('min/max: {}/{}'.format(np.min(xtrue), np.max(xtrue)))
                    self.data['xtrue'].append(xtrue)
                elif filename.endswith(self.mat_format):
                    img = sio.loadmat(os.path.join(dirpath, filename))['x']
                    for j in range(img.shape[-1]):
                        res_img = cv2.resize(np.squeeze(img[:,:,j]), 
                                             dsize=(self.N, self.N))
                        xtrue = np.array(res_img) / np.max(res_img)
                        print('min/max: {}/{}'.format(np.min(xtrue), np.max(xtrue)))
                        self.data['xtrue'].append(xtrue)
                else:
                    continue
    def gen_sys(self):
        def A(x): return self.scalefact * pad_fft(x, self.M, self.N, self.K, self.L)
        def At(x): return self.scalefact * unpad_ifft(x, self.M, self.N, self.K, self.L)
        return A, At
    def gen_data(self, A, At):
        for i in range(len(self.data['xtrue'])):
            xtrue_cated = holocat(vec(self.data['xtrue'][i]), self.ref)
            y = abs2(A(xtrue_cated)) + self.b
            y_pos = np.random.poisson(y)
            y_pos_gau = y_pos + np.random.normal(0, self.sigma**2, len(y_pos))
            print('average count: ', np.mean(y_pos_gau))
            print('maximum count: ', np.max(y_pos_gau))
            self.data['ynoisy'].append(y_pos_gau)
            x0_rand = holocat(vec(np.random.randn(self.N**2)), self.ref)
            def power_func(x): return At(np.multiply(np.divide(y_pos_gau, y_pos_gau+1), A(x)))
            x0_spectral = power_iter(power_func, x0_rand, 50)
            scale_x0 = np.sqrt(np.dot(y_pos_gau.clip(min=0), abs2(A(x0_spectral)))) / (norm(A(x0_spectral), 4)**2)
            x0_spectral = np.abs(scale_x0 * x0_spectral)[:self.N**2]
            self.data['x0'].append(x0_spectral)
            
if __name__ == "__main__":
    datapath = '../data/Set12'
    N = 128         
    scalefact = 0.02
    sigma = 1
    dataset = PrDataset(path=datapath, N=N, scalefact=scalefact, sigma=sigma)
    
        