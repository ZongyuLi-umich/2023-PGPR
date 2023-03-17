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

def main(parampath = '../config/params.txt', model = None, exp_to_do = []):
    parser = configparser(path=parampath)
    args = parser.parse_args()
    print('config args: ', args)

    dataset = PrDataset(path=args.datadir, 
                        N=args.imgsize, 
                        scalefact=args.scaleSYS, 
                        sigma=args.sigma)
    
    for i in range(min(args.Nimgs, len(dataset.data['xtrue']))):
        
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
            init_data = sio.load(init_path)
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
            alg_name = 'pnp_pgadmm'
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
                rho = 5
                scale = 1
                opt_pnppgadmm_scale = False
                opt_pnppgadmm_rho = False
                
                if opt_pnppgadmm_scale:
                    algoHandle = lambda scale: run_alg(alg=alg_name, 
                                                    sigma=args.sigma, 
                                                    delta=args.delta, 
                                                    niter=50, 
                                                    model=model,
                                                    scale = scale,
                                                    rho = rho,
                                                    verbose = False,
                                                    **kwargs)
                    
                    scale = optimizeTau(kwargs['xtrue'], algoHandle, [0, 2], maxfun=10)
                    print(f'opt scale = {scale}')
                
                
                if opt_pnppgadmm_rho:
                    algoHandle = lambda rho: run_alg(alg=alg_name, 
                                                    sigma=args.sigma, 
                                                    delta=args.delta, 
                                                    niter=50, 
                                                    model=model,
                                                    scale = scale,
                                                    rho = rho,
                                                    verbose = False,
                                                    **kwargs)
                    
                    rho = optimizeTau(kwargs['xtrue'], algoHandle, [0, 10], maxfun=10)
                    print(f'opt rho = {rho}')
                    

                    
                print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
                xout, cout = run_alg(alg=alg_name, 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=50, 
                                    model=model, 
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
    from model2 import DnCNN
    from collections import OrderedDict
    exp_to_do = ['pois', 'pnp_pgadmm'] #['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm']
    
    ##################################################
    # Reproducibility
    ##################################################
    init_env(seed_value=42)
    
    ##################################################
    # statistic
    ##################################################
    dnn_dict = {'dncnn': DnCNN}

    ##################################################
    # load config
    ##################################################
    with open('/n/higgins/z/xjxu/projects/2023-PGPR/config/pnp_config.json') as File:
        config = json.load(File)
        
    ##################################################
    # init the gpu usages
    ##################################################
    gpu_ids = config['settings']['gpu_ids']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##################################################
    # get rObj 
    ##################################################
    # load
    method_config = config['methods']['denoise']
    dnn_name = method_config['dnn_name']
    dnn = dnn_dict[dnn_name](config['networks'][dnn_name])
    dnn.to(device)

    # restore
    sgm = method_config['sgm']
    model_path = method_config['model_path'][f'sgm_{sgm}']
    checkpoint = torch.load(model_path)['model_state_dict']
    try: 
        name = 'dnn.'
        new_state_dict = OrderedDict({k[len(name):]: v for k, v in checkpoint.items()}) 
        dnn.load_state_dict(new_state_dict,strict=True)
    except: 
        print("Model cannnot load")
    
    ############################################################
    # run
    ############################################################
    with torch.no_grad():
        main(parampath = '../config/params.txt', model = dnn, exp_to_do = exp_to_do)
