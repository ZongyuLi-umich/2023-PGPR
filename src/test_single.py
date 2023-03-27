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


def test_single(i, root_result_dir, dataset, args = {}, 
                model_pnp = None, model_score=None,
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
                                        niter=args.gau_niter, 
                                        **kwargs)
            result = {'xout': xout, 'cout': cout}
            sio.savemat(f'{exp_path}', result)
        print(f'nrmse of {alg_name}: ', cout[-1])
        transcript.stop()
        xout_gau = xout
        
    
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
                                niter=args.pois_niter, 
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
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
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
            xout, cout = run_alg(alg=alg_name, 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pg_niter, 
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
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
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
            xout, cout = run_alg(alg=alg_name, 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pgTV_niter, 
                                reg1=args.regTV, 
                                reg2=0.1, 
                                **kwargs)
            result = {'xout': xout, 'cout': cout}
            sio.savemat(f'{exp_path}', result)
        print(f'nrmse of {alg_name}: ', cout[-1])
        transcript.stop()
        
    if 'pnp_pgadmm' in  exp_to_do: 
        # hyper parameters
        scale = args.pgADMM_scale
        rho = args.pgADMM_rho
        opt_pnppgadmm_scale = False
        opt_pnppgadmm_rho = False
        desp  = '_uiter3_muF'
            
        alg_name = 'pnp_pgadmm'
        exp_dir = f'{results_dir}/{alg_name}'
        check_and_mkdir(exp_dir)
        # copytree_code(f'{project_root}/src', exp_dir + '/')
        exp_path = f'{exp_dir}/result.mat'
        
        transcript.start(exp_dir + '/logfile.log', mode='a')
        print('\n###########################################################')
        print(f'{alg_name}')
        print('###########################################################')
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
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
            xout, cout = run_alg(alg=alg_name, 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pgADMM_niter, 
                                model=model_pnp, 
                                scale = scale,
                                rho = rho,
                                **kwargs)
            result = {'xout': xout, 'cout': cout}
            sio.savemat(f'{exp_path}', result)
        print(f'nrmse of {alg_name}: ', cout[-1])
        transcript.stop()
    
    if 'pnp_pgprox' in  exp_to_do: 
        # hyper parameters
        scale = args.pgPROX_scale
        rho   = args.pgPROX_rho
        opt_pnppgprox_scale = False
        desp  = ''
            
        alg_name = 'pnp_pgprox'
        exp_dir = f'{results_dir}/{alg_name}'
        check_and_mkdir(exp_dir)
        # copytree_code(f'{project_root}/src', exp_dir + '/')
        exp_path = f'{exp_dir}/result.mat'
        
        transcript.start(exp_dir + '/logfile.log', mode='a')
        print('\n###########################################################')
        print(f'{alg_name}')
        print('###########################################################')
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')
        else:
            return NotImplementedError
            
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
                                niter=args.pgPROX_niter, 
                                model=model_pnp, 
                                scale = scale,
                                rho = rho,
                                **kwargs)

            result = {'xout': xout, 'cout': cout}
            sio.savemat(f'{exp_path}', result)
        print(f'nrmse of {alg_name}: ', cout[-1])
        transcript.stop()
    
    if 'pnp_pgred' in  exp_to_do: 
        # hyper parameters
        scale = args.pgRED_scale
        rho   = args.pgRED_rho
        opt_pnppgred_scale = False
        desp  = ''
            
        alg_name = 'pnp_pgred'
        exp_dir = f'{results_dir}/{alg_name}'
        check_and_mkdir(exp_dir)
        # copytree_code(f'{project_root}/src', exp_dir + '/')
        exp_path = f'{exp_dir}/result.mat'
        
        transcript.start(exp_dir + '/logfile.log', mode='a')
        print('\n###########################################################')
        print(f'{alg_name}')
        print('###########################################################')
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')
        else:
            return NotImplementedError
            
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
                                niter=args.pgRED_niter, 
                                model=model_pnp, 
                                scale = scale,
                                rho = rho,
                                **kwargs)

            result = {'xout': xout, 'cout': cout}
            sio.savemat(f'{exp_path}', result)
        print(f'nrmse of {alg_name}: ', cout[-1])
        transcript.stop()
    
    if 'pnp_pgred_noise2self' in  exp_to_do: 
        # hyper parameters
        scale = args.pgRED_scale
        rho   = args.pgRED_rho
        opt_pnppgred_scale = False
        desp  = ''
            
        alg_name = 'pnp_pgred_noise2self'
        exp_dir = f'{results_dir}/{alg_name}'
        check_and_mkdir(exp_dir)
        # copytree_code(f'{project_root}/src', exp_dir + '/')
        exp_path = f'{exp_dir}/result.mat'
        
        transcript.start(exp_dir + '/logfile.log', mode='a')
        print('\n###########################################################')
        print(f'{alg_name}')
        print('###########################################################')
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
            print(f'[Old]: re-init x0 from result_pois.')
        else:
            return NotImplementedError
            
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
                                niter=args.pgRED_niter, 
                                model=model_pnp, 
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
            
        if args.init == 'Gaussian':
            kwargs['x0'] = xout_gau
            print(f'[Old]: re-init x0 from result_gau.')
        elif args.init == 'Poisson':
            kwargs['x0'] = xout_pois
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
            xout, cout = run_alg(alg=alg_name, 
                                        sigma=args.sigma, 
                                        delta=args.delta, 
                                        niter=args.pgSCORE_niter, 
                                        model=model_score, 
                                        **kwargs)
            result = {'xout': xout, 'cout': cout}
            sio.savemat(f'{exp_path}', result)
        print(f'nrmse of {alg_name}: ', cout[-1])
        transcript.stop()