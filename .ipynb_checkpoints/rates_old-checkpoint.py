from astro_constants import *
from source_parameters import *

from scipy.integrate import quad
from scipy import special
import numpy as np
#import matplotlib.pyplot as plt
#import os
import math
from tqdm import tqdm

def PIC_KN_tot(Ee,eps,nph):
    '''
    This function returns the total power loss of electrons of energy Ee 
    due to the process of inverse Compton scattering in a target photon field 
    defined by the arrays eps, nph.
    The calculation is done folowing the formulation of 
    Blumenthal G. R., Gould R. J., 1970, Reviews of Modern Physics, 42, 237
    Arguments:
    Ee  : [erg] energy of electrons  (vector).
    eps : [erg] energy of the target photons (vector).
    nph : [cm^{-3} erg^{-1}] number density of photons at the energy eps (vector). 
    '''
    
    def F(eps,nph,Ee):
        Gamma = 4. * eps * Ee / mec2**2
        ## Minimum and maximum energy bounds for the scattered photons
        e1min = eps
        e1max = Gamma * Ee / (1. + Gamma)
        dle1 = 0.025 # step size for the logarithmic energy range
        
        ## Logarithmic energy range for the scattered photons.
        le1 = np.arange(np.log10(e1min/eV), np.log10(e1max/eV), dle1) #log(E/eV)
        ## Scattered photon energies
        e1 = 10**le1 * eV
        ## Differences in scattered photon energies
        de1 = np.diff(e1)
        
        CC =  2 * np.pi * re**2 * me**2 * c**5 
        
        Sum = 0.
        for j in range(np.size(de1)):        
            q = e1[j]/(Gamma*(Ee-e1[j]))
            ## Kernel function describing the differential cross-section
            Fq = 2.*q* np.log(q) + (1+2*q)*(1-q) +.5*(1-q)*(Gamma*q)**2/(1+Gamma*q)
            ## Accumulates the contributions to the total power loss
            Sum = Sum + de1[j] * (e1[j] - eps) * CC / Ee**2 * nph / eps * Fq

        return Sum
    
    ## Differences in photon energies
    deps = np.diff(eps)
    
    tm1 =  np.zeros_like(Ee)
    SSum = np.zeros_like(Ee)
    
    print('\n\n Calculating IC rate (accounting for KN regime):')
    
    ## Iterates over electron energies (Ee)
    for j in tqdm( range(np.size(tm1)) ):
        ## Iterates over photon energy differentials (deps)
        for i in range(np.size(deps)):
            SSum[j] = SSum[j] + F(eps[i],nph[i],Ee[j]) * deps[i] 
        
        #print('    PIC( Ee=%1.3e eV )'%(Ee[j]/eV), end='')
        #print('\r',end='')
    
    print('')
    
    ## Accumulates the power loss contributions from each photon energy to the total power loss for each electron energy
    ## Total power loss for each electron energy due to inverse Compton scattering.
    
    return SSum





def rate_pg_cool(Ep, eps, nph):
    '''
    This function returns the coooling rate [s^[-1]] of protons of energy 
    Ep, due to the photo-pion process.
    The calculation is done folowing the formulation of
    Atoyan A. M., Dermer C. D., 2003, ApJ, 586, 79
    Arguments:
    Ep  : [erg] energy of protons  (vector).
    eps : [erg] energy of the target photons (vector).
    nph : [cm^{-3} erg^{-1}] number density of photons at the energy eps (vector).

    '''

    
    gp = Ep / mpc2  
    deps = np.diff(eps)
    dleps = np.log10( eps[1] / eps[0] )
    Ieps = np.zeros_like(gp)
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
        
        if( x<(200*MeV) ):
            return 0.
        elif( (x>= 200*MeV) and (x<500*MeV)  ):
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
    
    
    
    
    

## Photomeson

def sigma_pgamma(epsilon_r):
    """Simplified approach to calculate the cross-section.
    Eq.(41) of Khiali et al. 2015."""
    if 300*MeV <= epsilon_r <= 500*MeV:
        return 340e-30  # 340 microbarn in cm^2 if 300 MeV <= eps <= 500 MeV
    elif epsilon_r > 500*MeV:
        return 120e-30  # 120 microbarn in cm^2 if eps > 500 MeV
    else:
        return 0
    
def K_pgamma(epsilon_r):
    """Simplified approach to calculate the inelasticity of the interaction.
    Eq.(42) of Khiali et al. 2015."""
    if 300*MeV <= epsilon_r <= 500*MeV:
        return 0.2  # 0.2 if 300 MeV <= eps <= 500 MeV
    elif epsilon_r > 500*MeV:
        return 0.6  # 0.6 if eps > 500 MeV
    else:
        return 0
    
def n_ph(E_ph):
    """Isotropic photon field density.
    Eq. (4) from Mbarek et al. 2023."""
    r_c = 10 * R_s  # [cm] coronal size
    L_x = 7 * 10**43 # [erg s-1]
    epsilon_0 = 7e3 * erg # [erg] 7 keV = 7e3 * erg to eV compton hump energy
    
    U_x = L_x / (4 * np.pi * c * r_c**2) # [erg cm-3]

    n_x = U_x * ( epsilon_0**(-2) * (20*keV - 1*keV) + ( -(200*keV)**(-1) + (20*keV)**(-1) ) )
    print(f'\nPhoton field Energy Density for X-rays = {n_x:.5E} cm-3')
    # [cm-3]
    return n_E_ph
    

def t_pgamma(n, E):
    '''Eq.(40) of Khiali et al. 2015'''
    E_th = 145 * 1e6 * erg # [erg] 1 MeV = 1 * 1e6 * erg per eV - Photomeson production threshold for photon energies
    gamma_p = E / (m_e*c*c)
    
    
    lower_Eph = E_th / (2 * gamma_p)
    
    def integrand_Eph(E_ph):
        
        def integrand_epsr(epsilon_r):
            return sigma_pgamma(epsilon_r) * K_pgamma(epsilon_r) * epsilon_r
        
        # epsilon_r: photon energy in the rest frame of the proton
        lower_epsr = E_th
        upper_epsr = 2 * E_ph * gamma_p
        
        epsilon_r = np.linspace(20 * 1e3 * erg, 500 * 1e6 * erg, 100000)  # 20 keV a 500 MeV
        integral_epsr = np.trapz(integrand_epsr(epsilon_r), epsilon_r)
        
        return (n_ph(E_ph) / E_ph**2) * integral_epsr
    
    integral_Eph, _ = quad(lambda E_ph: integrand_Eph(E_ph), lower_Eph, 1e12)
    
    return c / (2 * gamma_p**2) * integral_Eph



    
    
    
# ## Romero 2010a

# def tshock(B,E):
#     """Implements Eq. (10) of Romero et al. (2010a)"""
#     eta = 0.1 # Efficiency of the acceleration [adim]

#     return eta*e*c*B/E

# def tsyn(B,E,m):
#     """Implements Eq. (12) of Romero et al. (2010a)"""
#     return (4/3)*((m_e/m)**3)*sigma_T*c*B*B/(m_e*c*c*8*np.pi) * (E/(m*c*c))

# def tbre(n,Z,E):
#     """Implements Eq. (17) of Romero et al. (2010a)"""

#     return 4*n*Z*Z*r_0*r_0*alpha_f*c*(np.log(2*E/(m_e*c*c))-1/3)

# def taccrece(E,lacc,B,rho):
#     """Implements Eq. (7) of Khiali et al. (2015)"""
#     va0 = B/(4*np.pi*rho)**0.5
#     Gamma = 1/(2**0.5)
#     va = va0*Gamma
#     t0 = lacc/va # Alfvén time [s]
#     E0 = m_e*c*c # * 6.241509e11 # erg to ev

#     return 1.3e5 *  ((m_p/m_e)**(0.5)) * ((E/E0)**(-0.21735284)) * 1/t0




## Hadronic Acceleration by Reconnection and Shock

def tacc_p_rec(E, lacc, B, rho):
    '''Eq. (6) of Khiali et al. (2015)'''
    va0 = B / np.sqrt(4 * np.pi * rho)
    Gamma = 1 / np.sqrt(2)
    va = va0 * Gamma # [cm s-1] Alfvén velocity
    t0 = lacc / va # [s] Alfvén time
    E0 = mp*c*c # [erg]

    return 1.3e5 * (E/E0)**(-0.1) * t0**(-1)
    

def tacc_shock(B, E):
    '''Eq.(9) of Khiali et al. 2015 / Eq. (10) of Romero et al. (2010a)'''
    eta = 0.1 # Efficiency of the acceleration [adim]

    return (eta*qe*c*B) / E


def t_pp(n, E):
    '''Eq. (19) of Khiali et al. (2015)'''
    L = np.log(E/ (1*TeV) )
    E_th = 280*MeV # Proton threshold kinetic energy for neutral pion production
    sigma_pp = (34.3 + 1.88*L + 0.25*L**2) * (1 - (E_th/E)**4 )**2 * 1e-27 # [cm2] 1 millibarn = 1e-27 cm^2
    k_pp = 0.5 # Total inelasticity of the process
    
    return n*c*sigma_pp*k_pp



## electrons

## Leptonic Acceleration by Reconnection and Shock

def tacc_e_rec(E, lacc, B, rho):
    '''Eq. (7) of Khiali et al. (2015)'''
    va0 = B / np.sqrt(4 * np.pi * rho)
    Gamma = 1 / np.sqrt(2)
    va = va0 * Gamma # [cm s-1] Alfvén velocity
    t0 = lacc / va # [s] Alfvén time
    E0 = mp*c*c # [erg]

    return 1.3e5 * np.sqrt(mp/me) * ((E/E0)**(-0.1)) * t0**(-1)


## Leptonic Radiative Losses

def tloss_syn(B, E, m):
    '''Eq. (12) of Romero et al. (2010a)'''
    
    return (4/3)*((me/m)**3)*sigmaT*c*B*B / (me*c*c*8*np.pi) * (E/(m*c*c))

def tloss_bre(n, Z, E):
    '''Eq. (17) of Romero et al. (2010a)'''
    
    return 4*n*Z*Z*re*re*alpha_fine*c*(np.log(2*E / (me*c*c)) - 1/3)
