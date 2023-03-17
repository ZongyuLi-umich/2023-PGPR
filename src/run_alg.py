# run_alg.py
# return selected algorithm
from Wirtinger_flow_huber_TV import *
from Wirtinger_flow_pois_gau import *
from Wirtinger_flow_score import *
from pnp_pgadmm import *

def run_alg(alg, sigma, delta, niter, reg1=0, reg2=0, model=None, scale = 1, verbose=True, rho=5, **kwargs):
    if alg == 'gau':
        xout, cout = Wintinger_flow_huber_TV(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       niter=niter, 
                                       gradhow='gau', 
                                       sthow='fisher', 
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
                                       niter=niter, 
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
                                       sigma=sigma, 
                                       delta=delta, 
                                       niter=niter, 
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
                                       sigma=sigma, 
                                       delta=delta, 
                                       niter=niter, 
                                       reg1=reg1, 
                                       reg2=reg2, 
                                       xtrue=kwargs['xtrue'],
                                       verbose = verbose)
    elif alg == 'pnp_pgadmm':
        xout, cout = pnp_pgadmm(A=kwargs['A'], 
                                At=kwargs['At'], 
                                y=kwargs['y'], 
                                b=kwargs['b'], 
                                x0=kwargs['x0'], 
                                ref=kwargs['ref'], 
                                sigma=sigma, 
                                delta=delta, 
                                niter=niter, 
                                rho = rho,
                                uiter= 1, 
                                xtrue=kwargs['xtrue'], 
                                model = model,
                                verbose = verbose)
                
    elif alg == 'pg_score':
        xout, cout = Wintinger_flow_score(A=kwargs['A'], 
                                       At=kwargs['At'], 
                                       y=kwargs['y'], 
                                       b=kwargs['b'], 
                                       x0=kwargs['x0'], 
                                       ref=kwargs['ref'], 
                                       sigma=sigma, 
                                       delta=delta, 
                                       niter=niter, 
                                       xtrue=kwargs['xtrue'], 
                                       model = model,
                                       verbose = verbose)
    else:
        return NotImplementedError
    return xout, cout