from astro_constants import *
from source_parameters import *

from scipy.integrate import quad
from scipy import special
import numpy as np

#import matplotlib.pyplot as plt
#import os
# import math
# from tqdm import tqdm


############ acceleration

def time_acc_regime3(va, delta_RX):
    v_in = 0.05 * va
    beta_in = v_in / c
    d_ur = 2 * beta_in * ( 3 * beta_in**2 + 3*beta_in + 1 ) / ( 3 * (beta_in + 0.5) * (1 - beta_in**2) )
    
    return 4 * delta_RX / (c * d_ur)

def timeacc_drift(E, B, va):
    '''Eq.(7) of Del Valle, de Gouveia Dal Pino & Kowal 2016 e de Gouveia Dal Pino & Kowal 2015'''
    v_rec = 0.05 * va
    
    return E / (qe*B*v_rec)



############ losses

## leptons

def rate_synch_e(B, E, m):
    '''Eq. (12) of Romero et al. (2010a)'''
    
    return (4/3)*((me/m)**3)*sigmaT*c*B*B / (me*c*c*8*np.pi) * (E/(m*c*c))

def rate_SSC_e():
    
    return

def rate_bremss_e(n, Z, E):
    '''Eq. (17) of Romero et al. (2010a)'''
    
    return 4*n*Z*Z*re*re*alpha_fine*c*(np.log(2*E / (me*c*c)) - 1/3)


## hadrons

def rate_synch_p(B, E):
    '''Eq. (12) of Romero et al. (2010a)'''
    
    return (4/3)*((me/mp)**3)*sigmaT*c*B*B / (me*c*c*8*np.pi) * (E/(mp*c*c))

def rate_p_p(n, E):
    '''Eq. (19) of Khiali et al. (2015)'''
    L = np.log(E/ (1*TeV) )
    E_th = 280*MeV # Proton threshold kinetic energy for neutral pion production
    sigma_pp = (34.3 + 1.88*L + 0.25*L**2) * (1 - (E_th/E)**4 )**2 * 1e-27 # [cm2] 1 millibarn = 1e-27 cm^2
    k_pp = 0.5 # Total inelasticity of the process
    
    return n*c*sigma_pp*k_pp

def rate_p_gamma():
    
    return

def rate_bethe_heitler():
    
    return