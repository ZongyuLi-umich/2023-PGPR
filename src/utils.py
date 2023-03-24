import h5py
import numpy as np
import matplotlib.pyplot as plt
import configargparse
import os
import torch
import shutil
import random

###############################################################################
# (Xiaojian's start)
###############################################################################
def check_and_mkdir(path):
    if not os.path.exists(path):
        # os.mkdir(path)
        os.makedirs(path)

def init_env(seed_value=42):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

def copytree_code(src_path, save_path):
    max_code_save = 100
    for i in range(max_code_save):
        code_path = save_path + 'code%d/' % i
        if not os.path.exists(code_path):
            shutil.copytree(src=src_path, dst=code_path)
            break
  
from scipy.optimize import fminbound
def optimizeTau(x, algoHandle, taurange, maxfun=20):
    evaluateSNR = lambda x, xhat: 20 * np.log10(
        np.linalg.norm(x.flatten('F')) / np.linalg.norm(x.flatten('F') - xhat.flatten('F')))
    fun = lambda tau: -evaluateSNR(x, algoHandle(tau)[0])
    tau = fminbound(fun, taurange[0], taurange[1], xtol=1e-3, maxfun=maxfun, disp=3)
    return tau
###############################################################################
# (Xiaojian's end)
###############################################################################

def getPatch(image, row, col, psize):
    #print(image.size())
    B = image.size(dim=0)
    x = image[:,row:row + psize, col:col + psize]
    # print(image.size())
    # print(row)
    # print(col)
    Gc = x.reshape((B, psize**2))
    return Gc

def GcT(myvec, row, col, psize, imsize, thing):
    B = myvec.size(dim=0)
    #thing = torch.zeros((B, imsize, imsize))
    thing[:,row:row+psize, col:col+psize] = thing[:,row:row+psize, col:col+psize] + myvec.reshape((-1, psize, psize)).cuda()
    #thing[:,row:row+psize, col:col+psize] = myvec.reshape((-1, psize, psize))
    return thing

def findScale(X, Y):
    a1 = sum(sum(np.multiply(X, X)))
    a2 = sum(sum(X))
    a3 = a2
    a4 = len(X) * len(X[0])
    b1 = sum(sum(np.multiply(X,Y)))
    b2 = sum(sum(Y))
    rhs = np.array([[a1, a2], [a3, a4]])
    lhs = np.array([b1, b2])
    out = np.linalg.solve(rhs, lhs)
    const = out[1] 
    lin = out[0]
    return lin * X + const

def config_parser():
    parser = configargparse.ArgumentParser(default_config_files=['/n/newberry/v/jashu/scoreMatchingFull/config/params.txt'])
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--logsdir", type=str, default='../logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--traindatadir", type=str, default='../logs/',
                        help='where to store training data')
    parser.add_argument("--valdatadir", type=str, default='../logs/',
                        help='where to store validation data')
    parser.add_argument("--datadir", type=str, default='../data/',
                        help='where to load training/testing data')

    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--patchsize", type=int, default=3,
                        help='size of each patch')
    parser.add_argument("--overlap", type=int, default=1,
                        help='overlap between patches')
    parser.add_argument("--sigma", type=float, default=0.05,
                        help='noise level')

    parser.add_argument("--nepochs", type=int, default=200,
                        help='number of epochs for training')

    parser.add_argument("--lr", type=float, default=0.001,
                        help='learning rate')
    parser.add_argument("--optim", type=str, default='sgd',
                        help='optimizer')
    parser.add_argument("--batchsize", type=int, default=1000,
                        help='batch size')
    parser.add_argument("--validfrac", type=float, default=0.2,
                        help='fraction of validation data')


    parser.add_argument("--do_online_test", type=bool, default=False,
                        help='if do testing while training')

    parser.add_argument("--online_test_epoch_gap", type=int, default=1,
                        help='the gap between each online test')
    
    parser.add_argument("--loss_fun", type=str, default='l2',
                        help='training loss')
    
    parser.add_argument("--gpu_ids", type=str, default='0',
                        help='the index of the gpu to use')

    parser.add_argument("--srcdir", type=str, default='../src/',
                        help='where the code is')
    
    return parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def snr(x, xhat):
    return 20 * np.log10(np.linalg.norm(x.flatten()) / np.linalg.norm(xhat.flatten()-x.flatten()))

def snr3(xi, y):
    #Matlab implementation redone
    r = np.sqrt(np.sum(xi**2)) / np.sqrt(np.sum(y**2))
    return 20*np.log10(r)

def snr2(signal, noisy_signal):
    mean_signal = np.mean(signal)
    signal_diff = signal - mean_signal
    var_signal = np.sum(np.mean(signal_diff**2))          ## variance of orignal data

    noise = noisy_signal - signal
    mean_noise = np.mean(noise)
    noise_diff = noise - mean_noise
    var_noise = np.sum(np.mean(noise_diff**2))            ## variance of noise

    if var_noise == 0:
        snr = 100                                       ## clean image
    else:
        snr = (np.log10(var_signal/var_noise))*10       ## SNR of the data
    return snr
    
class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    print(args)