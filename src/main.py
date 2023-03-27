import json
from model2 import DnCNN, Denoise
from test_all import *
from model2 import Unet

if __name__ == "__main__":
    ##################################################
    # Settings 
    ##################################################
    img_to_do     = [16,17,18,19]
    exp_to_do     = [] #['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 'pnp_pgred', 'pnp_pgred_noise2self', 'pg_score']
    dataset_name  = 'virusimg'
    project_root  = '/home/lizongyu/PycharmProjects/2023-PGPR'
    # params_config = f'{project_root}//src/config/params_{dataset_name}.txt'
    config  = f'{project_root}/src/config/config.json'
    
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
    dataset_pnp_config = pnp_config[dataset_name]
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
    model_score_path = allconfig['methods']['score'][dataset_name]['model_path']
    model_score = Unet(dim=allconfig['networks']['unet']['dim'])
    model_score.load_state_dict(torch.load(model_score_path, map_location='cpu'))
    model_score.to(device)
    print('model score # of params: ', count_parameters(model_score))

    ######################### load expargs ##########################
    args = Dict2Class(allconfig['expargs'])
    print('savedir: ', args.savedir)
    # model.eval()

    ############################################################
    # run
    ############################################################
    # with torch.no_grad():
    test_all(args=args, model_pnp=model_pnp, model_score=model_score,
            exp_to_do=exp_to_do, img_to_do=img_to_do)
            