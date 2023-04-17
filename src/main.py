import json
from model2 import DnCNN, Denoise
from test_all import *
from model2 import Unet

def run_config(config):
    ##################################################
    # Settings 
    ##################################################
    # img_to_do     = []
    # exp_to_do     = ['pois', 'pg_score'] #['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 'pnp_pgred', 'pnp_pgred_noise2self', 'pg_score']
    # dataset_name  = 'natureimg'
    # project_root  = '/home/lizongyu/PycharmProjects/2023-PGPR/src'
    # params_config = f'{project_root}//src/config/params_{dataset_name}.txt'
    
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
    with open(config) as File:
        allconfig = json.load(File)
        
    ######################### load expargs ##########################
    args = Dict2Class(allconfig['expargs'])
    print('savedir: ', args.savedir)
    copytree_code(os.getcwd(), f'{args.savedir}/src/')
    ##################################################
    # init the gpu usages
    ##################################################
    gpu_ids = allconfig['settings']['gpu_ids']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##################################################
    # get model 
    ##################################################
    # load config for pnp
    pnp_config  = allconfig['methods']['denoise']
    dataset_pnp_config = pnp_config[args.dataset_name]
    dnn_pnp_name       = dataset_pnp_config['dnn_name']
    sgm_pnp_name       = dataset_pnp_config['sgm_name']
    model_pnp_path     = dataset_pnp_config['model_path'][sgm_pnp_name]

    # restore model for pnp
    dnn_pnp   = dnn_dict[dnn_pnp_name](allconfig['networks'][dnn_pnp_name])
    model_pnp = Denoise(None, dnn_pnp, allconfig)
    
    checkpoint_pnp = torch.load(model_pnp_path, map_location='cpu')['model_state_dict']
    model_pnp.load_state_dict(checkpoint_pnp, strict=True)
    model_pnp.to(device)
    print('model pnp # of params: ', count_parameters(model_pnp))
    
    ########### load model for score function ############
    model_score_path = allconfig['methods']['score'][args.dataset_name]['model_path']
    model_score = Unet(dim=allconfig['networks']['unet']['dim'])
    model_score.load_state_dict(torch.load(model_score_path, map_location='cpu'))
    model_score.to(device)
    print('model score # of params: ', count_parameters(model_score))

    ########### load model for ddpm ############
    model_ddpm_path = allconfig['methods']['ddpm'][args.dataset_name]['model_path']
    model_ddpm = Unet(dim=allconfig['networks']['unet']['dim'])
    model_ddpm.load_state_dict(torch.load(model_ddpm_path, map_location='cpu'))
    model_ddpm.to(device)
    print('model ddpm # of params: ', count_parameters(model_ddpm))
    
    # model.eval()

    ############################################################
    # run
    ############################################################
    # with torch.no_grad():
    test_all(args=args, model_pnp=model_pnp, model_score=model_score, 
             model_ddpm=model_ddpm, exp_to_do=args.exp_to_do, img_to_do=args.img_to_do)   

if __name__ == "__main__":
    directory  = '/home/lizongyu/PycharmProjects/2023-PGPR/src/config'
    # json_files = []
    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         json_files.append(f)
    
    # for i in range(len(json_files)):
    #     run_config(json_files[i])
        
    # config1 = 'config-0.020-sigma-0.5.json'
    # config2 = 'config-0.020-sigma-0.75.json'
    config3 = 'config-0.020-sigma-1.25.json'
    config4 = 'config-0.020-sigma-1.5.json'
    # run_config(os.path.join(directory, config1))
    # run_config(os.path.join(directory, config2))
    run_config(os.path.join(directory, config3))
    run_config(os.path.join(directory, config4))