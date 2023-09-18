from utils import cal_eval_from_true
if __name__ == "__main__":
    root_result_dir = "../result/density_small_test"
    datadir = '/home/lizongyu/PycharmProjects/2023-PGPR/data/density_small/density_small_test_old.mat'
    img_to_do = [0,1,2]
    exp_to_do = ['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 'pnp_pgred', 'pnp_pgred_noise2self', 'pg_ddpm', 'pg_score_apg']
    # make summary
    cal_eval_from_true(root_result_dir, data_dir=datadir, img_to_do=img_to_do, exp_to_do=exp_to_do)    
    