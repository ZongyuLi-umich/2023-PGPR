# run_alg.py
# return selected algorithm
from Wirtinger_flow_huber_TV import *
from Wirtinger_flow_pois_gau import *
from pnp_pgadmm import *

def run_alg(alg, sigma, delta, niter, reg1, reg2, **kwargs):
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
                                       xtrue=kwargs['xtrue'])
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
                                       xtrue=kwargs['xtrue'])
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
                                       xtrue=kwargs['xtrue'])
    elif alg == 'pg_pnp':
        return NotImplementedError
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
                                       xtrue=kwargs['xtrue'])
    elif alg == 'pg_score':
        return NotImplementedError
    else:
        return NotImplementedError
    return xout, cout