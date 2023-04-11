# run_alg.py
# return selected algorithm
from Wirtinger_flow_huber_TV import *
from Wirtinger_flow_pois_gau import *
from Wirtinger_flow_score import *
from pnp_pgadmm import *
from pnp_pgprox import *
from pnp_pgred import *
from pnp_pgred_noise2self import *
from Wirtinger_flow_ddpm import *


def run_alg(alg, args, model_pnp = None, model_score=None, model_ddpm=None, 
            verbose=True, **kwargs):
    if alg == 'gau':
        xout, cout = Wintinger_flow_huber_TV(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       niter=args.gau_niter, 
                                       gradhow='gau', 
                                       sthow='lineser', 
                                       reg1=0, 
                                       reg2=0, 
                                       xtrue=kwargs['xtrue'],
                                       verbose = verbose)
    elif alg == 'pois':
        xout, cout = Wintinger_flow_huber_TV(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       niter=args.pois_niter, 
                                       gradhow='pois', 
                                       sthow='fisher', 
                                       reg1=0, 
                                       reg2=0, 
                                       xtrue=kwargs['xtrue'],
                                       verbose = verbose)
    elif alg == 'pg':
        xout, cout = Wintinger_flow_pois_gau(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=args.sigma, 
                                       delta=args.delta, 
                                       niter=args.pg_niter, 
                                       reg1=0, 
                                       reg2=0, 
                                       xtrue=kwargs['xtrue'],
                                       verbose = verbose)

    elif alg == 'pg_tv':
        xout, cout = Wintinger_flow_pois_gau(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=args.sigma, 
                                       delta=args.delta, 
                                       niter=args.pgTV_niter, 
                                       reg1=args.regTV, 
                                       reg2=0.1, 
                                       xtrue=kwargs['xtrue'],
                                       verbose = verbose)
    elif alg == 'pnp_pgadmm':
        xout, cout = pnp_pgadmm(A=kwargs['A'], 
                                At=kwargs['At'], 
                                y=kwargs['y'], 
                                b=kwargs['b'], 
                                x0=kwargs['x0'], 
                                ref=kwargs['ref'], 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pgADMM_niter, 
                                xtrue=kwargs['xtrue'], 
                                model = model_pnp,
                                mu = None,
                                scale = args.pgADMM_scale,
                                rho = args.pgADMM_rho,
                                uiter= 3, 
                                verbose = verbose)
    elif alg == 'pnp_pgprox':
        xout, cout = pnp_pgprox(A=kwargs['A'], 
                                At=kwargs['At'], 
                                y=kwargs['y'], 
                                b=kwargs['b'], 
                                x0=kwargs['x0'], 
                                ref=kwargs['ref'], 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pgPROX_niter, 
                                xtrue=kwargs['xtrue'], 
                                model = model_pnp,
                                mu = None,
                                scale = args.pgPROX_scale, 
                                rho = args.pgPROX_rho,
                                verbose = verbose)
    elif alg == 'pnp_pgred':
        xout, cout = pnp_pgred(A=kwargs['A'], 
                                At=kwargs['At'], 
                                y=kwargs['y'], 
                                b=kwargs['b'], 
                                x0=kwargs['x0'], 
                                ref=kwargs['ref'], 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pgRED_niter, 
                                xtrue=kwargs['xtrue'], 
                                model = model_pnp,
                                mu = None,
                                scale = args.pgRED_scale, 
                                rho = args.pgRED_rho,
                                verbose = verbose)
    elif alg == 'pnp_pgred_noise2self':
        xout, cout = pnp_pgred_noise2self(A=kwargs['A'], 
                                At=kwargs['At'], 
                                y=kwargs['y'], 
                                b=kwargs['b'], 
                                x0=kwargs['x0'], 
                                ref=kwargs['ref'], 
                                sigma=args.sigma, 
                                delta=args.delta, 
                                niter=args.pgRED_niter, 
                                xtrue=kwargs['xtrue'], 
                                model = model_pnp,
                                mu = None,
                                scale = args.pgRED_scale, 
                                rho = args.pgRED_rho,
                                verbose = verbose)                            
                        
    elif alg == 'gau_score':
        xout, cout = Wintinger_flow_score(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=args.sigma, 
                                       delta=args.delta, 
                                       niter=args.pgSCORE_niter, 
                                       xtrue=kwargs['xtrue'], 
                                       model = model_score,
                                       gradhow = 'gau',
                                       verbose = verbose)
    elif alg == 'pois_score':
        xout, cout = Wintinger_flow_score(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=args.sigma, 
                                       delta=args.delta, 
                                       niter=args.pgSCORE_niter, 
                                       xtrue=kwargs['xtrue'], 
                                       model = model_score,
                                       gradhow = 'pois',
                                       verbose = verbose)
    elif alg == 'pg_score':
        xout, cout = Wintinger_flow_score(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=args.sigma, 
                                       delta=args.delta, 
                                       niter=args.pgSCORE_niter, 
                                       xtrue=kwargs['xtrue'], 
                                       model = model_score,
                                       gradhow = 'pg',
                                       verbose = verbose)
    elif alg == 'gau_ddpm':
        xout, cout = Wirtinger_flow_score_ddpm(A=kwargs['A'], 
                                            At=kwargs['At'], 
                                            y=kwargs['y'], 
                                            b=kwargs['b'], 
                                            x0=kwargs['x0'], 
                                            ref=kwargs['ref'], 
                                            sigma=args.sigma, 
                                            delta=args.delta, 
                                            niter=args.pgDDPM_niter, 
                                            xtrue=kwargs['xtrue'], 
                                            model = model_ddpm,
                                            gradhow = 'gau',
                                            verbose = verbose)
    elif alg == 'pois_ddpm':
        xout, cout = Wirtinger_flow_score_ddpm(A=kwargs['A'], 
                                            At=kwargs['At'], 
                                            y=kwargs['y'], 
                                            b=kwargs['b'], 
                                            x0=kwargs['x0'], 
                                            ref=kwargs['ref'], 
                                            sigma=args.sigma, 
                                            delta=args.delta, 
                                            niter=args.pgDDPM_niter, 
                                            xtrue=kwargs['xtrue'], 
                                            model = model_ddpm,
                                            gradhow = 'pois',
                                            verbose = verbose)
    elif alg == 'pg_ddpm':
        xout, cout = Wirtinger_flow_score_ddpm(A=kwargs['A'], 
                                            At=kwargs['At'], 
                                            y=kwargs['y'], 
                                            b=kwargs['b'], 
                                            x0=kwargs['x0'], 
                                            ref=kwargs['ref'], 
                                            sigma=args.sigma, 
                                            delta=args.delta, 
                                            niter=args.pgDDPM_niter, 
                                            xtrue=kwargs['xtrue'], 
                                            model = model_ddpm,
                                            gradhow = 'pg',
                                            verbose = verbose)
    else:
        return NotImplementedError
    return xout, cout