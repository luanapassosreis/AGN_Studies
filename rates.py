from astro_constants import *
from source_parameters import *

from scipy.integrate import quad
from scipy import special
import numpy as np
import math

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

def rate_synch_e(B, Ee):
    '''Eq. (5) of Romero et al. (2010)'''
    UB = B**2 / (8*np.pi)
    return 4/3 * sigmaT * c * UB * Ee / (mec2)**2
    # return 4/3 * sigmaT * c * UB * (Ee/mec2)**2

def rate_SSC_e():
    
    return

def rate_bremss_e(n, Z, Ee):
    '''Eq. (17) of Romero et al. (2010a)'''
    
    return 4 * n * Z**2 * re**2 * alpha_fine * c * (np.log(2*Ee / (mec2)) - 1/3)


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

def rate_p_gamma(Ep, eps, nph):
    Eth_pg = 145 * 1e6 * eV
    k1 = 0.2
    s1 = 340 * 1e-6 * barn
    k2 = 0.6
    s2 = 120 * 1e-6 * barn
    C1 = -.5*k1*s1*(200*MeV)**2
    C2 = -.5*k2*s2*(500*MeV)**2 + .5*k1*s1*(500*MeV)**2 - .5*k1*s1*(200*MeV)**2
    
    def II(eps,Ep):
        
        if( (2*eps*Ep /mpc2) < (200*MeV) ):
            return 0.
        
        elif( ( (200*MeV)<=(2*eps*Ep /mpc2) ) & ( (2*eps*Ep /mpc2)<(500*MeV) ) ):
            return k1*s1/2 *(2*eps*Ep/mpc2)**2 + C1
        
        elif( (2*eps*Ep/mpc2) > 500*MeV ):
            return k2*s2/2 *(2*eps*Ep/mpc2)**2 + C2
    
    SSum = np.zeros(np.size(Ep))
    deps = np.diff(eps)

    for j in range(np.size(Ep)):
        for i in range(np.size(deps)):
            if(eps[i] > (Eth_pg * mpc2 / 2 / Ep[j]) ):
                SSum[j] = SSum[j] + mp**2 * c**5 / 2 / Ep[j]**2 * deps[i] * nph[i] / eps[i]**2 * II(eps[i], Ep[j])                    
    return SSum


def rate_pg_cool(Ep, eps, nph):
    
    gp = Ep / mpc2  
    deps = np.diff(eps)
    dleps = np.log10( eps[1] / eps[0] )
    Ieps = np.zeros_like(gp) + small
    epsth = 145*MeV 
    
    sigma1 = 340 * 1e-6*barn
    K1 = 0.2
    sigma2 = 120 * 1e-6*barn
    K2 = 0.6
    
    def fI(eps, gp):
        x = 2* eps * gp
        def I1(x):
            return sigma1*K1 * 0.5* x**2
        def I2(x):
            return sigma2*K2 * 0.5* x**2
        
        if( x < (200*MeV) ):
            return 0.
        elif( (x >= 200*MeV) and (x < 500*MeV)  ):
            return I1(x) - I1(200*MeV)
        elif( x>= 500*MeV ):
            return I1(500*MeV) - I1(200*MeV) + I2(x) - I2(500*MeV)
    
    for i in range(len(gp)):
        eps_inf = epsth / (2*gp[i] )
        
        if( (eps_inf >= eps[0] ) and ( eps_inf < eps[len(deps)] ) ):
            j0 = int( math.ceil( np.log10( eps_inf / eps[0] ) / dleps ) )
            for j in range(j0,len(deps)):
                Ieps[i] = Ieps[i] + deps[j]*nph[j] / (eps[j]**2) * fI(eps[j], gp[i])
    
    return c / (2 * gp**2) * Ieps 


def rate_bethe_heitler(Ep, eps, nph):
    
    gp = Ep / mpc2
    deps = np.diff(eps)
    dleps = np.log10( eps[1] / eps[0] )
    Ieps = np.zeros_like(gp) + small
    
    ## de dn_x/de e^2 
    ## e sigma e de
    
    
    
    return c / (2 * gp**2) * Ieps