# test_all.py
from email import utils

from utils2 import *
import numpy as np
from config_parser import *
from run_alg import *
from dataloader import *
import sys
# sys.path.insert(0, '/n/newberry/v/jashu/scoreMatchingFull/src')
from model2 import Unet
from utils import *
import scipy.io as sio
import transcript
import os, time, datetime

def main(parampath = './config/params.txt', model = None, exp_to_do = [], img_to_do=[], project_root=''):
    parser = configparser(path=parampath)
    args = parser.parse_args()
    print('config args: ', args)

    dataset = PrDataset(path=args.datadir, 
                        N=args.imgsize, 
                        scalefact=args.scaleSYS, 
                        sigma=args.sigma)
    
    for i in img_to_do:
        
        #  make folders
        results_dir = f'{args.savedir}_imgsize{args.imgsize}_sf{args.scaleSYS}_sigma{args.sigma}/{i}'
        check_and_mkdir(results_dir)
        
        # load init data
        exp_dir = f'{results_dir}/init'
        check_and_mkdir(exp_dir)
        transcript.start(exp_dir + '/logfile.log', mode='a')
        
        print('\n###########################################################')
        print(f'init : {exp_dir}')
        print('###########################################################')
        init_path = f'{exp_dir}/result.mat'
        try: 
            init_data = sio.loadmat(init_path)
            print(f'[Old]: init data  loaded from {init_path}.')
        except:
            print(f'[New]: init data  running to {init_path}.')
            init_data  = {'ynoisy': dataset.data['ynoisy'][i],
                          'x0':dataset.data['x0'][i],
                          'xtrue': dataset.data['xtrue'][i]}
            sio.savemat(f'{init_path}', init_data)
        transcript.stop()
        
        # init parameters
        kwargs = {'A': dataset.A, 
                  'At': dataset.At, 
                  'y': init_data['ynoisy'].squeeze(),
                  'b': dataset.b,
                  'ref': dataset.ref,
                  'x0': init_data['x0'].squeeze(),
                  'xtrue': init_data['xtrue'].squeeze(),
                  }
        
        if 'gau' in  exp_to_do:
            alg_name = 'gau'
            exp_dir = f'{results_dir}/{alg_name}'
            check_and_mkdir(exp_dir)
            exp_path = f'{exp_dir}/result.mat'
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')
                xout, cout = run_alg(alg=alg_name, 
                                            sigma=0, 
                                            delta=0, 
                                            niter=200, 
                                            **kwargs)
                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
                

        if 'pois' in  exp_to_do:  
            alg_name = 'pois'
            exp_dir = f'{results_dir}/{alg_name}'
            check_and_mkdir(exp_dir)
            exp_path = f'{exp_dir}/result.mat'
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=0, 
                                    delta=0, 
                                    niter=200, 
                                    **kwargs)
                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
            xout_pois = xout
            
        if 'pg' in  exp_to_do: 
            alg_name = 'pg'
            exp_dir = f'{results_dir}/{alg_name}'
            check_and_mkdir(exp_dir)
            exp_path = f'{exp_dir}/result.mat'
            
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')

            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=10, 
                                    **kwargs)
                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
            
        if 'pg_tv' in  exp_to_do: 
            alg_name = 'pg_tv' 
            exp_dir = f'{results_dir}/{alg_name}'
            check_and_mkdir(exp_dir)
            exp_path = f'{exp_dir}/result.mat'
            
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')

            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=50, 
                                    reg1=args.regTV, 
                                    reg2=0.1, 
                                    **kwargs)
                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
            
        
        if 'pnp_pgadmm' in  exp_to_do: 
            # hyper parameters
            scale = 0.5
            rho = 32
            opt_pnppgadmm_scale = False
            opt_pnppgadmm_rho = False
            desp  = '_uiter3_muF'
            
            alg_name = 'pnp_pgadmm'
            exp_dir = f'{results_dir}/{alg_name}/{sgm_name}_scale{scale}_rho{rho}/{str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}{desp}'
            check_and_mkdir(exp_dir)
            copytree_code(f'{project_root}/src', exp_dir + '/')
            exp_path = f'{exp_dir}/result.mat'
            
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')
            
            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:                
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=30, 
                                    model=model, 
                                    scale = scale,
                                    rho = rho,
                                    **kwargs)
                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
            
        if 'pnp_pgprox' in  exp_to_do: 
            # hyper parameters
            scale = 0.5
            rho   = 0.5
            opt_pnppgprox_scale = False
            desp  = ''
            
            alg_name = 'pnp_pgprox'
            exp_dir = f'{results_dir}/{alg_name}/{sgm_name}_scale{scale}_rho{rho}/{str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}{desp}'
            check_and_mkdir(exp_dir)
            copytree_code(f'{project_root}/src', exp_dir + '/')
            exp_path = f'{exp_dir}/result.mat'
            
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')
            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                ############################################
                # run
                ############################################
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=20, 
                                    model=model, 
                                    scale = scale,
                                    rho = rho,
                                    **kwargs)

                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
            
        if 'pnp_pgred' in  exp_to_do: 
            # hyper parameters
            scale = 0.5
            rho   = 100
            opt_pnppgred_scale = False
            desp  = ''
            
            alg_name = 'pnp_pgred'
            exp_dir = f'{results_dir}/{alg_name}/{sgm_name}_scale{scale}_rho{rho}/{str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}{desp}'
            check_and_mkdir(exp_dir)
            copytree_code(f'{project_root}/src', exp_dir + '/')
            exp_path = f'{exp_dir}/result.mat'
            
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')
            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                ############################################
                # run
                ############################################
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=20, 
                                    model=model, 
                                    scale = scale,
                                    rho = rho,
                                    **kwargs)

                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
            
        if 'pg_score' in  exp_to_do: 
            alg_name = 'pg_score'    
            exp_dir = f'{results_dir}/{alg_name}'
            check_and_mkdir(exp_dir)
            exp_path = f'{exp_dir}/result.mat'
            
            transcript.start(exp_dir + '/logfile.log', mode='a')
            print('\n###########################################################')
            print(f'{alg_name}')
            print('###########################################################')
            
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')

            try: 
                result = sio.loadmat(exp_path)
                xout = result['xout'].squeeze()
                cout  = result['cout'].squeeze()
                print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
            except:
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                            sigma=args.sigma, 
                                            delta=args.delta, 
                                            niter=20, 
                                            model=model, 
                                            **kwargs)
                result = {'xout': xout, 'cout': cout}
                sio.savemat(f'{exp_path}', result)
            print(f'nrmse of {alg_name}: ', cout[-1])
            transcript.stop()
        
if __name__ == "__main__":
    import json
    from model2 import DnCNN, Denoise
    
    ##################################################
    # Settings 
    ##################################################
    img_to_do     = [0]
    exp_to_do     = ['pois', 'pnp_pgadmm'] #['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 'pnp_pgred']
    dataset_name  = 'virusimg'
    project_root  = '/n/higgins/z/xjxu/projects/2023-PGPR'
    params_config = f'{project_root}//src/config/params_{dataset_name}.txt'
    pnp_config    = f'{project_root}/src/config/pnp_config.json'
    
    ##################################################
    # reproducibility
    ##################################################
    init_env(seed_value=42)
    
    ##################################################
    # statistic
    ##################################################
    dnn_dict = {'dncnn': DnCNN}

    ##################################################
    # load config
    ##################################################
    with open(pnp_config) as File:
        config = json.load(File)
        
    ##################################################
    # init the gpu usages
    ##################################################
    gpu_ids = config['settings']['gpu_ids']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##################################################
    # get model 
    ##################################################
    # load config
    method_config  = config['methods']['denoise']
    dataset_config = method_config[dataset_name]
    dnn_name       = dataset_config['dnn_name']
    sgm_name       = dataset_config['sgm_name']
    model_path     = dataset_config['model_path'][sgm_name]

    # restore model
    dnn   = dnn_dict[dnn_name](config['networks'][dnn_name])
    model = Denoise(None, dnn, config)
    
    checkpoint = torch.load(model_path)['model_state_dict']
    model.load_state_dict(checkpoint,strict=True)
    model.to(device)
    # model.eval()

    ############################################################
    # run
    ############################################################
    with torch.no_grad():
        main(parampath=params_config, model=model, exp_to_do=exp_to_do, img_to_do=img_to_do, project_root=project_root)
        