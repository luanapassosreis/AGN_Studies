from numba import jit
from constants import *
from rateFn import *
from scipy.integrate import quad
from scipy import special
import numpy as np 
#import matplotlib.pyplot as plt
import os
import math
# from tqdm import tqdm

    

@jit(nopython=True)
def ICKerr(Ee, eps, EIC):
    GG = 4*eps*Ee / mec2**2
    q = EIC /( GG*(Ee - EIC) )
    Fq = 2*q*np.log(q)  + (1+2*q)*(1-q) + 0.5*(1-q)*(GG*q)**2/( 1 + GG*q ) 
    
    return Fq / Ee**2 / eps


@jit(nopython=True)
def dN_dtdEdV2_IC(EIC,Ee,Ne,eps,nph):
    
    deps = np.diff(eps)
    dEe = np.diff(Ee)
    AlEe = np.log10( Ee[1] / Ee[0] )
    Aleps = np.log10( eps[1] / eps[0] )
    
    qIC = np.zeros(len(EIC))
    #print("    Calculating IC q-emissivity ...")
    for i in range(len(EIC)):
        epsmaxa = EIC[i]
        jmaxa =  np.log10( epsmaxa / eps[0] )  / Aleps
        jend = min(len(deps), int(jmaxa) )
        
        for j in range(0,jend):
            Eemina = EIC[i] / 2 * ( 1 + np.sqrt( 1 + (mec2)**2/EIC[i]/eps[j] ) )
            kmina = np.log10( Eemina / Ee[0] )  / AlEe 
            k0 = max( 0, int( kmina + 1) )
            if( k0>len(dEe)):
                k0 = len(dEe)
            
            for k in range(k0, len(dEe)):
                qIC[i] = qIC[i] + deps[j]*nph[j]*dEe[k]*Ne[k] * ICKerr(Ee[k],eps[j],EIC[i])
    
    qIC = 3*sigmaT*c*mec2**2 / 4 * qIC
    
    return qIC

@jit(nopython=True)
def ahaG(x):
    
    C0 = 1.808*x**(1/3) / np.sqrt( 1 + 3.4*x**(2/3) )
    P1 = 1 + 2.21*x**(2/3) + 0.347* x**(4/3)
    P2 = 1 + 1.353*x**(2/3) + 0.217* x**(4/3)
    return C0 * P1 / P2 * np.exp(-x) 

@jit(nopython=True)
def dP_dEdV_synAha(E_syn,E,N,B,mass):

    def pavPsynAha(E,Eph,B):
        '''
        analytic approximation for the  sincle particles
        synchrotron power per unit ENERGY, as reported 
        average over the pitch angle, within 0.2%.
        This approximation is derived by Aharonian et al. 2010.
        '''
        Ec = 3 *qe*h*B /(4*np.pi*mass*c) * (E/mass/c**2)**2
        x = Eph / Ec
        
        return np.sqrt(3) * qe**3 * B / (mass*c**2 * h) * ahaG(x) 
    
    
    dE = np.diff(E) 
    Emit_syn = np.zeros(len(E_syn))
    #Emit_syn = Emit_syn 
       
    #print('\n    Calculating synchrotron emission ...')
    for k in range(len(E_syn)):
        for l in range(len(dE)):
            Emit_syn[k] = Emit_syn[k]  + pavPsynAha(E[l],E_syn[k],B) * N[l] *dE[l]
    
    return Emit_syn