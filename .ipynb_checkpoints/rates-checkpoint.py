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



## apagar

def Ppsyn_tot(Ep,B):
    '''
    equivalent to equation (5) of Romero et al. 2010
    '''
    
    UB = B**2/(8*np.pi)
    return 4/3 * (me / mp)**2 * sigmaT * c * UB * (Ep/mpc2)**2



def Psyn_tot(Ee,B):
    '''
    equivalent to equation (5) of Romero et al. 2010
    '''
    
    UB = B**2/(8*np.pi)
    return 4/3 * sigmaT * c * UB * (Ee/mec2)**2


############ losses

## leptons

def rate_synch_e(B, Ee, m):
    '''Eq. (5) of Romero et al. (2010)'''
    UB = B**2 / (8*np.pi)
    return 4/3 * sigmaT * c * UB * Ee / (mec2)**2
    # return 4/3 * sigmaT * c * UB * (Ee/mec2)**2

def rate_SSC_e():
    
    return

def rate_bremss_e(n, Z, E):
    '''Eq. (17) of Romero et al. (2010a)'''
    
    return 4*n*Z*Z*re*re*alpha_fine*c*(np.log(2*E / (me*c*c)) - 1/3)


## hadrons

def rate_synch_p(B, Ep):
    '''Eq. (5) of Romero et al. (2010)'''
    UB = B**2 / (8*np.pi)
    return (4/3) * (me/mp)**3 * sigmaT * c * UB * Ep / (mec2 * mpc2)
    # return (4/3) * (me / mp)**2 * sigmaT * c * UB * (Ep/mpc2)**2

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