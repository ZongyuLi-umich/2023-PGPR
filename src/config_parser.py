# config_parser.py
import configargparse

def configparser(path='../config/params.txt'):
     parser = configargparse.ArgumentParser(default_config_files=[path])
     parser.add_argument('--config', is_config_file=True,
                        help='config file path')
     parser.add_argument("--datadir", type=str, default='../data/',
                        help='where to load training/testing data')
     parser.add_argument("--imgsize", type=int, default=128,
                        help='image size')
     parser.add_argument("--scaleSYS", type=float, default=0.02,
                        help='scale factor applied to the system matrix')
     parser.add_argument("--sigma", type=float, default=1,
                        help='noise variance')
     parser.add_argument("--delta", type=float, default=5,
                        help='error bound to estimate the infinite sum')
     parser.add_argument("--regTV", type=float, default=2,
                        help='regularization parameter for TV')
     return parser
     
                        