# test_all.py
from email import utils

from utils2 import *
from run_alg import *
from dataloader import *
import sys
# sys.path.insert(0, '/n/newberry/v/jashu/scoreMatchingFull/src')
from utils import *
import scipy.io as sio
import transcript


def run_wo_init(results_dir, alg_name, model_pnp, model_score, 
                model_ddpm, args, kwargs):
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
        xout, cout = run_alg(alg=alg_name, args=args, model_pnp = model_pnp, 
                             model_score=model_score, model_ddpm=model_ddpm, 
                            verbose=True, **kwargs)
        result = {'xout': xout, 'cout': cout}
        sio.savemat(f'{exp_path}', result)
    print(f'nrmse of {alg_name}: ', cout[-1])
    transcript.stop()


def run_with_init(results_dir, alg_name, model_pnp, model_score, 
                  model_ddpm, args, kwargs):
    exp_dir = f'{results_dir}/{alg_name}'
    check_and_mkdir(exp_dir)
    exp_path = f'{exp_dir}/result.mat'
    
    transcript.start(exp_dir + '/logfile.log', mode='a')
    print('\n###########################################################')
    print(f'{alg_name}')
    print('###########################################################')
        
    if args.init == 'Gaussian':
        kwargs['x0'] = sio.loadmat(f'{results_dir}/gau/result.mat')['xout'].squeeze()
        print(f'[Old]: re-init x0 from result_gau.')
    elif args.init == 'Poisson':
        kwargs['x0'] = sio.loadmat(f'{results_dir}/pois/result.mat')['xout'].squeeze()
        print(f'[Old]: re-init x0 from result_pois.')
    else:
        return NotImplementedError

    try: 
        result = sio.loadmat(exp_path)
        xout = result['xout'].squeeze()
        cout  = result['cout'].squeeze()
        print(f'[Old]: result of [{alg_name}] loaded from {exp_path}.')
    except:
        print(f'[New]: result of [{alg_name}] running to save to {exp_path}.')  
        xout, cout = run_alg(alg=alg_name, args=args, model_pnp = model_pnp, 
                             model_score=model_score, model_ddpm=model_ddpm, 
                            verbose=True, **kwargs)
        result = {'xout': xout, 'cout': cout}
        sio.savemat(f'{exp_path}', result)
    print(f'nrmse of {alg_name}: ', cout[-1])
    transcript.stop()
    
    
def test_single(i, root_result_dir, dataset, args = {}, 
                model_pnp = None, model_score=None, model_ddpm=None,
                exp_to_do = []):
    #  make folders
    results_dir = f'{root_result_dir}/{i}'
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
    
    for alg_name in exp_to_do:
        if alg_name in ['gau', 'pois']:
            run_wo_init(results_dir, alg_name, model_pnp, 
                        model_score, model_ddpm, args, kwargs)
        elif alg_name in ['pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 
                     'pnp_pgred', 'pnp_pgred_noise2self', 'gau_score',
                     'pois_score', 'pg_score', 'gau_ddpm', 'pois_ddpm',
                     'pg_ddpm']:
            run_with_init(results_dir, alg_name, model_pnp, 
                        model_score, model_ddpm, args, kwargs)
        else:
            raise NotImplementedError
                