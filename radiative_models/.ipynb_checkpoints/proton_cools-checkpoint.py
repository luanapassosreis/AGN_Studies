import matplotlib
matplotlib.use('agg')
from constants import *
###python packages
from functools import partial
import numpy as np
import pandas as pd
import os
import h5py





def UniLogEGrid(Emin,Emax,DlE):
    '''
    This function return a vector of energies values 
    uniformly spaced in log10(E/eV)
    '''
    lE = np.arange(np.log10(Emin/eV), np.log10(Emax/eV), DlE)
    E = 10**lE * eV
    
    return E



def Ppsyn_tot(Ep,B):
    '''
    equivalent to equation (5) of Romero et al. 2010
    '''
    
    UB = B**2/(8*np.pi)
    return 4/3 * (me / mp)**2 * sigmaT * c * UB * (Ep/mpc2)**2



#@jit(nopython=True)
def rate_pg_cool(Ep, eps, nph):
    '''
    This cool function differs from the rate_pg_coll function
    in the K1, K1 factors. The rate_pg_coll function consider
    K1=K2 =1.
    '''
    
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
    
    
    
    
#@jit(nopython=True)
def rate_pg_coll(Ep, eps, nph):
    
    gp = Ep / mpc2
    deps = np.diff(eps)
    dleps = np.log10( eps[1] / eps[0] )
    Ieps = np.zeros_like(gp) + small
    epsth = 145*MeV 
    
    sigma1 = 340 * 1e-6*barn
    K1 = 1.
    sigma2 = 120 * 1e-6*barn
    K2 = 1.
    
    
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
        
        if( (eps_inf >= eps[0] ) and ( eps_inf < eps[len(deps)-1] ) ):
            j0 = int( math.ceil( np.log10( eps_inf / eps[0] ) / dleps ) )
            for j in range(j0,len(deps)):
                Ieps[i] = Ieps[i] + deps[j]*nph[j] / (eps[j]**2) * fI(eps[j], gp[i])
    
    return c / (2 * gp**2) * Ieps 




def sigmaBHe(eps):
    '''
    Paramerisation of Bethe-Heitler cross section
    according to Maximon (1968)
    J. Res. Nat. Bur. Stand., B, 72: 79-88(Jan.-Mar. 1968).
    DOI:https://doi.org/10.6028/jres.072B.011
    '''
    
    Z = 1 ##(only protons)
    if( (eps>=2) and (eps<=4) ):
        f1 = ( (eps - 2)/eps )**3
        
        eeta = (eps - 2) / (eps + 2)
        f2 = 1 + 0.5*eeta + (23/40)*eeta**2 + (37/120)*eeta**3 + (61/192)*eeta**4  
        
        return 2*np.pi / 3 * alpha_fine * re**2 * Z**2 * f1 * f2  
    
    if( eps>4 ):
        z3 = 1.20206
        g1 =  (28/9)* np.log(2*eps)
        g2 = - (218/27)
        g3 = (2/eps)**2
        g4 = 6*np.log(2*eps) - 7/2 + 2/3*( np.log(2*eps) )**3 - (np.log(2*eps) )**2 - (1/3)*np.pi**2 * np.log(2*eps) + 2*z3 + np.pi**2 /6
        g5 = (2/eps)**4 * ( 3/16 * np.log(2*eps) + 1/8 )
        g6 = - (2/eps)**6 * (29/9/256 * np.log(2*eps) - 77/27/512 )
        
        return alpha_fine * re**2 * Z**2 * ( g1 + g2 + g3*g4 + g5 + g6 )



    
def rate_BH_coll(Ep, eps, nph):
    
    def IIrest(ggp, eeps):
        
        I = 0.
        eps_pr_sup = 2* eeps * ggp
        ksup = int( math.ceil( np.log10( eps_pr_sup / eps_pr[0] ) / np.log10( eps_pr[1] / eps_pr[0] ) ) )
        for k in range(ksup -1 ):
            I = I + sigmaBHe(eps_pr[k]/mec2) * eps_pr[k] * deps_pr[k]
        
        return I
        
    
    gammp_max = Ep[len(Ep)-1] / mpc2
    eps_max =  eps[len(eps)-1]
    
    leps_pr = np.arange( np.log10(1.05*MeV / eV), np.log10(2*gammp_max*eps_max/eV), 0.05 )
    eps_pr = 10**(leps_pr) * eV
    deps_pr = np.diff(eps_pr)
    
    gp = Ep / mpc2
    deps = np.diff(eps)
    
    II = np.zeros_like(Ep)    
    print('\n\n Calculating Bethe-Heitler sec. electrons:')
#    for i in tqdm( range( len(Ep)-1 ) ):
    for i in range( len(Ep)-1 ):
                
        eps_inf = 1.05*MeV/(2*gp[i])
        
        if( (eps_inf >= eps[0] ) and ( eps_inf < eps[len(deps)-1] ) ):
                
            j0 = int( math.ceil( np.log10( eps_inf / eps[0] ) / np.log10( eps[1] / eps[0] ) ) )
            for j in range( j0, len(deps)):
                II[i] = II[i] +  deps[j] * nph[j] / (eps[j]**2) * IIrest(gp[i],eps[j])


    return c / (2*gp**2) * II





###  Example on how to define the input arrays (internal units are in cgs, see constants.py file) ###
#####################################################################################################
### AlE = 0.025
### Ep = UniLogEGrid(5*mpc2,1e18*eV, AlE)  # energy of protons [erg]  
###
### eps_min = 1e-2 *eV
### eps_max = 1e2 *eV
### eps = UniLogEGrid(eps_min, eps_max, AlE) # energy of target photons [erg ]  
### nph = photonf(eps)   ##  photonf eh a sua funcao que calcula o photon field density [cm^-3 erg^-1]
### Bj = 1 # [Gauss]


pRates = {}
    
pRates["Ep"] = Ep
pRates["synch"] = (1/Ep)*Ppsyn_tot(Ep,Bj)
pRates["photo-pion"] = rate_pg_cool(Ep,eps,nph)
pRates["B-H"] = (2*me/mp)*rate_BH_coll(Ep, eps, nph)


totalCoolRate = pRates["synch"] +  pRates["photo-pion"] + pRates["B-H"]
totalCoolTime = 1./totalCoolRate
