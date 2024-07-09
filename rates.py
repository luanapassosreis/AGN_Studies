from constants import *
from scipy.integrate import quad
from scipy import special
import numpy as np
#import matplotlib.pyplot as plt
#import os
import math
from tqdm import tqdm

def PIC_KN_tot(Ee,eps,nph):
    "
    This function returns the total power loss of electrons of energy Ee 
    due to the process of inverse Compton scattering in a target photon field 
    defined by the arrays eps, nph.
    The calculation is done folowing the formulation of 
    Blumenthal G. R., Gould R. J., 1970, Reviews of Modern Physics, 42, 237
    Arguments:
    Ee  : [erg ]is the energy of electrons  (vector).
    eps : [erg]energy of the target photons (vector).
    nph : [cm^{-3} erg^{-1}] number density of photons at the energy eps (vector). 
    
    "
    
    def F(eps,nph,Ee):
        Gamma = 4. * eps * Ee / mec2**2
        e1min = eps
        e1max = Gamma * Ee / (1. + Gamma)
        dle1 = 0.025
        le1 = np.arange(np.log10(e1min/eV), np.log10(e1max/eV), dle1) #log(E/eV)
        e1 = 10**le1 * eV
        de1 = np.diff(e1)
        
        CC =  2 * np.pi * re**2 * me**2 * c**5 
        
        Sum = 0.
        for j in range(np.size(de1)):        
            q = e1[j]/(Gamma*(Ee-e1[j]))
            Fq = 2.*q* np.log(q) + (1+2*q)*(1-q) +.5*(1-q)*(Gamma*q)**2/(1+Gamma*q)
            Sum = Sum + de1[j] * (e1[j] - eps) * CC / Ee**2 * nph / eps * Fq

        return Sum
    
    deps = np.diff(eps)
    
    tm1 =  np.zeros_like(Ee)
    SSum = np.zeros_like(Ee)
    print('\n\n Calculating IC rate (accounting for KN regime):')
    for j in tqdm( range(np.size(tm1)) ):
        for i in range(np.size(deps)):
            SSum[j] = SSum[j] + F(eps[i],nph[i],Ee[j]) * deps[i] 
        #print('    PIC( Ee=%1.3e eV )'%(Ee[j]/eV), end='')
        #print('\r',end='')
    print('')    
    
    
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
    
    


