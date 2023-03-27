# test_all.py
from utils2 import *
import numpy as np
from run_alg import *
from dataloader import *
import sys
# sys.path.insert(0, '/n/newberry/v/jashu/scoreMatchingFull/src')
from utils import *
import transcript
from joblib import Parallel, delayed
from test_single import test_single

    
def test_all(args = {}, model_pnp = None, model_score=None,
            exp_to_do = [], img_to_do=[]):
    
    # pass empty list to test on all images
    root_result_dir = f'{args.savedir}'
    check_and_mkdir(root_result_dir)
    transcript.start(root_result_dir + '/logfile.log', mode='a')
    print('config args: ', args.__dict__)

    dataset = PrDataset(path=args.datadir, 
                        N=args.imgsize, 
                        scalefact=args.scaleSYS, 
                        sigma=args.sigma)
    transcript.stop()
    
    if not img_to_do:
        img_to_do = range(len(dataset.data['xtrue']))
    
    if not exp_to_do:
        exp_to_do = ['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 'pnp_pgred', 'pnp_pgred_noise2self', 'pg_score']
    
    
    Parallel(n_jobs=args.ncore)(delayed(test_single)(i=i,
                                            root_result_dir=root_result_dir, 
                                            dataset=dataset, 
                                            args = args, 
                                            model_pnp = model_pnp, 
                                            model_score=model_score,
                                            exp_to_do=exp_to_do) for i in img_to_do)
            
        
    make_summary(root_result_dir, img_to_do, exp_to_do)
        
        
def make_summary(root_result_dir, img_to_do, exp_to_do):
    all_data = {}
    for algname in exp_to_do:
        all_data[algname] = np.zeros(len(img_to_do))
    for i, idx in enumerate(img_to_do):
        results_dir = f'{root_result_dir}/{idx}'
        for alg_name in exp_to_do:
            alg_dir = f'{results_dir}/{alg_name}'
            with open(f'{alg_dir}/logfile.log', 'r') as f:
                last_line = f.readlines()[-1]
                nrmse = last_line.split(' ')[-1]
                all_data[alg_name][i] = nrmse
    transcript.start(root_result_dir + '/summary.log', mode='a')
    print('###########################################################')
    print('summary result')
    print('###########################################################')
    for key in all_data.keys():
        print(f'alg name: {key} || mean: {np.mean(all_data[key]):.4f} || std: {np.std(all_data[key]):.4f} || max: {np.amax(all_data[key]):.4f} || min: {np.amin(all_data[key]):.4f}')
    transcript.stop()
            
    