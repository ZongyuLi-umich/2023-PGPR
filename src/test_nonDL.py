# test_nonDL.py
from utils import *
import numpy as np
from config_parser import *
from run_alg import *
from dataloader import *

def main(parampath = '../config/params.txt'):
    parser = configparser(path=parampath)
    args = parser.parse_args()
    print('config args: ', args)
    dataset = PrDataset(path=args.datadir, 
                        N=args.imgsize, 
                        scalefact=args.scaleSYS, 
                        sigma=args.sigma)
    all_results = {'xout_gau': [], 'cout_gau': [],
                   'xout_pois': [], 'cout_pois': [],
                   'xout_pg': [], 'cout_pg': [],
                   'xout_pgTV': [], 'cout_pgTV': []}
    for i in range(len(dataset.data['xtrue'])):
        if i == 1:
            break
        kwargs = {'A': dataset.A, 
                  'At': dataset.At, 
                  'y': dataset.data['ynoisy'][i],
                  'b': dataset.b,
                  'ref': dataset.ref,
                  'x0': dataset.data['x0'][i],
                  'xtrue': dataset.data['xtrue'][i],
                  }
        xout_gau, cout_gau = run_alg(alg='gau', 
                                    sigma=0, 
                                    delta=0, 
                                    niter=200, 
                                    reg1=0, 
                                    reg2=0, 
                                    **kwargs)
        print('nrmse of pois: ', cout_gau[-1])
        xout_pois, cout_pois = run_alg(alg='pois', 
                                    sigma=0, 
                                    delta=0, 
                                    niter=200, 
                                    reg1=0, 
                                    reg2=0, 
                                    **kwargs)
        print('nrmse of pois: ', cout_pois[-1])
        kwargs['x0'] = xout_pois
        xout_pg, cout_pg = run_alg(alg='pg', 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=50, 
                                    reg1=0, 
                                    reg2=0, 
                                    **kwargs)
        print('nrmse of pg: ', cout_pg[-1])
        xout_pgTV, cout_pgTV = run_alg(alg='pg_tv', 
                                    sigma=args.sigma, 
                                    delta=args.delta, 
                                    niter=50, 
                                    reg1=args.regTV, 
                                    reg2=0.1, 
                                    **kwargs)
        print('nrmse of pgTV: ', cout_pgTV[-1])
        
        all_results['xout_gau'].append(xout_gau)
        all_results['cout_gau'].append(cout_gau)
        all_results['xout_pois'].append(xout_pois)
        all_results['cout_pois'].append(cout_pois)
        all_results['xout_pg'].append(xout_pg)
        all_results['cout_pg'].append(cout_pg)
        all_results['xout_pgTV'].append(xout_pgTV)
        all_results['cout_pgTV'].append(cout_pgTV)
    
    results_dir = os.path.join('../result', args.expname)
    check_and_mkdir(results_dir)
    np.save(os.path.join(results_dir, 'xtrue-x0-ynoisy.npy'), dataset.data)
    np.save(os.path.join(results_dir, 'alg-results.npy'), all_results)
    
    
        
        
if __name__ == "__main__":
    main()