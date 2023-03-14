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
        xout_pois, cout_pois = run_alg(alg='pois', 
                                    sigma=0, 
                                    delta=0, 
                                    niter=200, 
                                    reg1=0, 
                                    reg2=0, 
                                    **kwargs)
        print('nrmse of pois: ', cout_pois[-1])
if __name__ == "__main__":
    main()