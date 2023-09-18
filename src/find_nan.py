from utils import find_nan
if __name__ == "__main__":
    root_result_dir = "../result/density_small_test"
    img_to_do = range(12)
    exp_to_do = ['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 
                     'pnp_pgred', 'pnp_pgred_noise2self', 
                     'pois_score_apg']
    find_nan(root_result_dir, img_to_do, exp_to_do)  