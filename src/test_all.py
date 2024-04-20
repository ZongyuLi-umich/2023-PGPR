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
             model_ddpm = None, exp_to_do = [], img_to_do=[]):
    
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
        exp_to_do = ['gau', 'gau_amp', 'gs', 'pois', 'pg', 'pg_tv', 
                     'pnp_pgadmm', 'pnp_pgprox', 
                     'pnp_pgred', 'pnp_pgred_noise2self', 
                     'gau_score', 'pois_score', 'pg_score',
                     'gau_score_apg', 'pois_score_apg', 'pg_score_apg',
                     'gau_ddpm', 'pois_ddpm', 'pg_ddpm']
    
    
    Parallel(n_jobs=args.ncore)(delayed(test_single)(i=i,
                                            root_result_dir=root_result_dir, 
                                            dataset=dataset, 
                                            args = args, 
                                            model_pnp = model_pnp, 
                                            model_score=model_score,
                                            model_ddpm = model_ddpm,
                                            exp_to_do=exp_to_do) for i in img_to_do)
            
    # make summary
    make_summary(root_result_dir, img_to_do, exp_to_do) 
        
        

            
    