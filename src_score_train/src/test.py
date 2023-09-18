from utils import *
from dataLoader import scoreloader
from model import Fern
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.io import savemat
import math

class scoretester():
    def __init__(self, model, 
                is_best = True, 
                verbose=False, is_denoise = False, im = [], #whole image, need to turn into patches
                args=''):
        super(scoretester, self).__init__()
        if args:
            self.args = args 
        else:
            parser = config_parser()
            self.args = parser.parse_args()
        #print('config args:', self.args)

        if is_denoise:
            width = len(im[0,:])
            tot_patches = ((width - self.args.patchsize) // self.args.overlap + 1) ** 2
            im_patches = np.zeros((tot_patches, self.args.patchsize**2))
            a = 0
            for row in range(0, width-self.args.patchsize+1, self.args.overlap):
                for col in range(0, width-self.args.patchsize+1, self.args.overlap):
                    mypatch = im[row:row+self.args.patchsize, col:col + self.args.patchsize]
                    im_patches[a,:] = mypatch.reshape(1, self.args.patchsize**2)
                    a = a+1

            self.loader = scoreloader(projdir=self.args.testdatadir,
                             psize = self.args.patchsize, mode = 'test', 
                             noisy_im = im_patches,
                             sigma = self.args.sigma)
        else:
            self.loader = scoreloader(projdir=self.args.testdatadir,
                             psize = self.args.patchsize,
                             sigma = self.args.sigma)
        
        self.psize, self.lsize = self.loader.getsize()

        self.testloader = data.DataLoader(self.loader,
                                           batch_size=self.args.batchsize,
                                           shuffle=False,
                                           num_workers=8,
                                           pin_memory=True)
        in_channels = self.psize**2
        self.model = Fern(D = self.args.netdepth, W = self.args.netwidth, input_ch = in_channels,
        output_ch = self.psize**2)

        self.model = model
        self.device = next(self.model.parameters()).device

    def test(self, dataset_key='test', denormalize = True):
        with torch.no_grad():
            results = []
            #GT = []
            self.model.eval()
            for batch in self.testloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                pred = self.model.forward(batch['input']*1)/1
                results.append(pred.cpu())
                #GT.append(batch['output'].cpu())

            # print(results[0:-1])
            results_stacked = torch.stack(results[0:-1]).reshape(-1)
            #GT_stacked = torch.stack(GT[0:-1]).reshape(-1)
            return torch.cat((results_stacked, results[-1].reshape(-1)), dim=0).numpy()#, \
                    #torch.cat((GT_stacked, GT[-1].reshape(-1)), dim=0).numpy()

if __name__ == "__main__":
    checkpoint_dir = "../logs/2023-01-23-16-23-26"
    tester = scoretester(f'{checkpoint_dir}/models')
    first, second = tester.test()
    print(first.shape)
    print(second.shape)
    mdic = {"first": first, "second": second}
    savemat("test_output.mat", mdic)