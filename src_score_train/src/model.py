import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from collections import OrderedDict
#from numpy.typing import ArrayLike, NDArray
from typing import *
from utils import * 

class Fern(nn.Module):
    def __init__(self, D=8, W=256, input_ch=9, input_ch_views = 0, output_ch=9, skips=None, imsize=64):
        super(Fern, self).__init__()
        if skips is None:
            skips = [D//2]
        self.D = D
        self.W = W 
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.input_ch_views = input_ch_views
        self.skips = skips 
        self.imsize = imsize
        self.psize = int(input_ch**0.5)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        self.output_linear = nn.Linear(W, output_ch)


    def forward(self, y):
        y = y.view(-1, self.imsize, self.imsize) #size is Bx64x64
        B = y.size(dim=0)
        out = torch.zeros_like(y) 
        totpatches = (self.imsize-self.psize+1)**2
        patchstack = torch.zeros(B*totpatches, self.psize**2).cuda()
        count = 0
        for a in range(self.imsize - self.psize):
            for b in range(self.imsize - self.psize):
                x = getPatch(y, a, b, self.psize) #size is big lol Bx9??
                patchstack[count*B:(count+1)*B, :] = x
                count = count + 1

        input_pts, input_views = torch.split(patchstack, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h) #size is B*totpatches by 9
        count = 0
        out = torch.zeros(B, self.imsize, self.imsize).cuda()
        for row in range(self.imsize-self.psize):
            for col in range(self.imsize-self.psize):
                out = GcT(outputs[B*count:B*count+B,:], row, col, self.psize, self.imsize, out)
                #out = out + GcT(outputs[B*count:B*count+B,:], row, col, self.psize, self.imsize)
                count = count + 1

        return out.reshape(-1, self.imsize**2)
    
    # def load_model(self, checkpoint_dir):
    #     model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    #     # try:
    #     #     self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    #     #     print('load pre-trained model successfully from: {}!'.format(checkpoint_dir))
    #     # except:
    #     #     raise IOError(f"load Checkpoint '{checkpoint_dir}' failed! ")

    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     model.to(device)
    #     return model


class Fern2(nn.Module):
    def __init__(self, D=8, W=256, input_ch=9, input_ch_views = 0, output_ch=9, skips=None):
        super(Fern, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(input_ch, W),
                        nn.LeakyReLU(),
                        nn.Linear(W, 2*W),
                        nn.LeakyReLU(),
                        nn.Linear(2*W, 4*W),
                        nn.LeakyReLU(),
                        nn.Linear(4*W, 8*W),
                        nn.LeakyReLU(),
                        nn.Linear(8*W, 8*W),
                        nn.LeakyReLU(),
                        nn.Linear(8*W, 4*W),
                        nn.LeakyReLU(),
                        nn.Linear(4*W, 2*W),
                        nn.Tanh(),
                        nn.Linear(2*W, W),
                        nn.Tanh(),
                        nn.Linear(W, output_ch),
                    )


    def forward(self, x):
        outputs = self.model(x)
        return outputs
    


if __name__ == "__main__":
    model = Fern(D=8, W=40, input_ch=9, output_ch=9, input_ch_views=0)
    print(model)

    # x = torch.randn(9)
    # y = model.forward(x)
    # print('y: ', y)
    # ytrue = torch.ones(9)
    # criterion = nn.MSELoss()
    # loss = criterion(y, ytrue)
    # print('loss: ', loss.item())
    # loss.backward()
