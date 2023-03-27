# Wirtinger_flow_huber_TV.py
import numpy as np
from utils2 import *
import torch
from eval_metric import *
from utils import *
import json
from model2 import Unet
from dataloader import *
from joblib import Parallel, delayed
from run_alg import run_alg

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
        
def Wintinger_flow_score(A, At, y, b, x0, ref, sigma, delta, 
                            niter, xtrue, model, iteration, 
                            sigma_max, sigma_min, scale, verbose=True):
    
    M = len(y)
    N = len(x0)
    out = []
    sn = np.sqrt(N).astype(int)
    out.append(nrmse(x0, xtrue))
    x = np.copy(x0)
    phi, grad_phi, fisher = get_grad(sigma, delta)

    Ax = A(holocat(x, ref))

    lastnrmse = 1
    T = iteration # 2-5
    # sigma_max: 0.02-0.08
    # sigma_min: 0.005-0.019
    # step size: 0.1-0.4
    sigmas = np.geomspace(sigma_max, sigma_min, niter)
    for iter in range(niter):
        lsize = 128
        for t in range(T):
            netinput = torch.from_numpy(np.reshape(x, (1,1,lsize,lsize))).float().cuda()
            network_sigma = torch.from_numpy(np.array([sigmas[iter]])).float().cuda()
            scorepart = -model.forward(netinput, network_sigma).cpu().detach().numpy().reshape((lsize, lsize))/sigmas[iter]
            scorepart = scorepart.reshape(-1)
            #scorepart = -model.forward(torch.from_numpy(x.reshape((lsize, lsize)))).cpu().detach().numpy()/0.05
            scorepart = np.squeeze(scorepart)
            #grad_f = np.real(At(grad_phi(Ax, y, b)))[:N] + reg1 * diff2d_adj(grad_huber_v(Tx, reg2), sn, sn)
            grad_f = np.real(At(grad_phi(Ax, y, b)))[:N] + 1*scorepart
            
            Adk = A(holocat(grad_f, np.zeros_like(grad_f)))
            D1 = np.sqrt(fisher(Ax, b))
            # mu = - (norm(grad_f)**2)/ (norm(np.multiply(Adk, D1))**2) * (sigmas[iter]/0.05)**2
            #mu = - (norm(grad_f)**2) / (norm(np.multiply(Adk, D1))**2 + reg1 * (norm(np.multiply(Tdk, D2))**2))
            mu = -scale*(sigmas[iter]**2)
            # mu = -(sigmas[iter]**2)/4
            x += mu * grad_f
            x[(x < 0)] = 0 # set non-negatives to zero
            Ax = A(holocat(x, ref))

        out.append(nrmse(x, xtrue))
        if np.abs(lastnrmse-out[-1]) < 1e-5:
            break
        lastnrmse = out[-1]
        if verbose: 
            print(f'iter: {iter:03d} / {niter:03d} || nrmse (out, xtrue): {out[-1]:.4f}')
    return x, out


def grid_search_single(i, kwargs, args, model_score):
    exp_dir = f'{args.savedir}/{i}'
    check_and_mkdir(exp_dir)
    transcript.start(exp_dir + '/logfile.log', mode='a')
    iteration = random.randint(2, 4) # 2-4
    sigma_max = random.randint(20, 100) / 1000 # 0.02-0.1
    sigma_min = random.randint(10, 190) / 10000 # 0.001-0.019
    scale = random.randint(100, 500) / 1000 # step size: 0.1-0.5
    print('###########################################################')
    print(f'T: {iteration} || sigma_max: {sigma_max:.4f} || sigma_min: {sigma_min:.4f} || scale: {scale:.4f}')
    print('###########################################################')
    xout, cout = Wintinger_flow_score(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=args.sigma, 
                                       delta=args.delta, 
                                       niter=args.pgSCORE_niter, 
                                       xtrue=kwargs['xtrue'], 
                                       model = model_score,
                                       iteration = iteration, 
                                       sigma_max = sigma_max, 
                                       sigma_min = sigma_min, 
                                       scale = scale,
                                       verbose = True)
    print(f'nrmse of {i}th experiment: ', cout[-1])
    
    transcript.stop()
    
def find_best_run(root_result_dir, num_trials):
    all_data = np.zeros(num_trials)
    for idx in range(num_trials):
        results_dir = f'{root_result_dir}/{idx}'
        with open(f'{results_dir}/logfile.log', 'r') as f:
            last_line = f.readlines()[-1]
            nrmse = last_line.split(' ')[-1]
            all_data[idx] = nrmse
    transcript.start(root_result_dir + '/summary.log', mode='a')
    best_param = np.argmin(all_data)
    print('###########################################################')
    print('summary result')
    print('###########################################################')
    print(f'best run: #{best_param}')
    transcript.stop()
    
if __name__ == "__main__":
    img_id   = 9
    dataset_name  = 'virusimg'
    project_root  = '/home/lizongyu/PycharmProjects/2023-PGPR'
    # params_config = f'{project_root}//src/config/params_{dataset_name}.txt'
    config  = f'{project_root}/src/config/config.json'
    
    ##################################################
    # reproducibility
    ##################################################
    init_env(seed_value=42)
    ##################################################
    # load config
    ##################################################
    with open(config) as File:
        allconfig = json.load(File)
        
    ##################################################
    # init the gpu usages
    ##################################################
    gpu_ids = allconfig['settings']['gpu_ids']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ########### load model for score function ############
    model_score_path = allconfig['methods']['score'][dataset_name]['model_path']
    model_score = Unet(dim=allconfig['networks']['unet']['dim'])
    model_score.load_state_dict(torch.load(model_score_path, map_location='cpu'))
    model_score.to(device)
    print('model score # of params: ', count_parameters(model_score))

    ######################### load expargs ##########################
    args = Dict2Class(allconfig['expargs'])
    print('savedir: ', args.savedir)
    
    root_result_dir = f'{args.savedir}'
    check_and_mkdir(root_result_dir)
    transcript.start(root_result_dir + '/logfile.log', mode='a')
    print('config args: ', args.__dict__)
    transcript.stop()
    
    dataset = PrDataset(path=args.datadir, 
                        N=args.imgsize, 
                        scalefact=args.scaleSYS, 
                        sigma=args.sigma)
    
    init_data  = {'ynoisy': dataset.data['ynoisy'][img_id],
                        'x0':dataset.data['x0'][img_id],
                        'xtrue': dataset.data['xtrue'][img_id]}
    # init parameters
    kwargs = {'A': dataset.A, 
                'At': dataset.At, 
                'y': init_data['ynoisy'].squeeze(),
                'b': dataset.b,
                'ref': dataset.ref,
                'x0': init_data['x0'].squeeze(),
                'xtrue': init_data['xtrue'].squeeze(),
                }
    # run poisson
    xout_pois, cout_pois = run_alg(alg='pois', 
                                sigma=0, 
                                delta=0, 
                                niter=args.pois_niter, 
                                **kwargs)
    kwargs['x0'] = xout_pois
    print(f'[Old]: re-init x0 from result_pois.')
    num_trials = 1000
    Parallel(n_jobs=args.ncore)(delayed(grid_search_single)(i=i,
                                            kwargs=kwargs, 
                                            args=args, 
                                            model_score = model_score
                                            ) for i in range(num_trials))
    find_best_run(root_result_dir, num_trials)

    
