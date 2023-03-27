from utils import make_summary
if __name__ == "__main__":
    root_result_dir = "../result/purple-poisson-init"
    img_to_do = range(20)
    exp_to_do = ['gau', 'pois', 'pg', 'pg_tv', 'pnp_pgadmm', 'pnp_pgprox', 'pnp_pgred', 'pnp_pgred_noise2self', 'pg_score']
    # make summary
    make_summary(root_result_dir, img_to_do, exp_to_do)    
    