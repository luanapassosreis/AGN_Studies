from numba import jit
from constants import *
import numpy as np
import os

def L_Edd(M):
    
    return 1.26e38 * (M / Msun)  
    
def StripJetPas(z, G8, lamb, a, f_lmin, MBH):

    '''
    This function receives the parameters being optimised and returns 
    the local jet physical properties (B_diss, G_diss, Pd_diss, z_pk, 
    z_diss, 0., sigmaBK) based on the reconnection striped jet model 
    of Giannios & Uzdenzky 2019
    '''
    
    @jit(nopython=True)
    def IGammZ(X, k):
    
        return (1-X)**(3-k) / (k-3) -  2*(1-X)**(2-k) / (k-2)  +  (1-X)**(1-k) / (k-1)
    
    @jit(nopython=True)
    def FX(X,X0,k,zeta):
    
        return IGammZ(X, k) - IGammZ(X0, k) - zeta
    
    
    @jit(nopython=True)
    def RootChi(X,k,zeta):
        #X0 = X[0]
        Xroot = 0.
        for i in range(len(X)-1):
            FXup = FX(X[i+1],X[0],k,zeta)
            FXon = FX(X[i],X[0],k,zeta)
            if( FXup*FXon  < 0. ):
                Xroot = 0.5*(X[i] + X[i+1])
                return Xroot
                break            
    
    DlEe = 0.02
    th_jet = 0.2 #sim
    z_diss = z * pc
 
    ### reconnection striped jet model:
    zeta_pk = 1./3.
    k = ( 3*a - 1 ) / ( 2*a - 2 )
    eps_rec = 0.1
        
    lmin = f_lmin * G * MBH / c**2
    X_min =  0.01                     
    X_max = 0.999
    dlX = 0.001
    lX = np.arange(np.log10(X_min), np.log10(X_max), dlX) #log(E/eV)
    X = 10**lX
      
    Lj = lamb * L_Edd(MBH)
    z_pk = lmin * G8**2 / ( 6 * eps_rec ) 

    zeta_diss = 2 * eps_rec * z_diss  / (lmin * G8**2)
    if zeta_diss < 0.:
        zeta_diss = abs(zeta_diss)
        print("LESS THAN ZERO!!!!")
   
    Xdiss     = RootChi(X,k,zeta_diss)
    if Xdiss == None:
        Xdiss = 0.5
        print(z, zeta_diss, f_lmin, G8)
    G_diss    = G8 * Xdiss
    
    denB_diss = G8**6 * ( th_jet * zeta_diss * lmin )**2 *np.pi * c
    B_diss2 = 4*np.pi * 4 * eps_rec**2 / denB_diss * ( 1. - Xdiss ) / Xdiss**2 * Lj
    B_diss = np.sqrt( B_diss2 )
    Pd_diss = (1 - Xdiss )**k / Xdiss**2 * zeta_diss * Lj  
    sigmaBK = (1 - Xdiss ) / Xdiss

    pasout = [B_diss, G_diss, Pd_diss, z_pk, z_diss, 0., sigmaBK]
    
    return pasout



