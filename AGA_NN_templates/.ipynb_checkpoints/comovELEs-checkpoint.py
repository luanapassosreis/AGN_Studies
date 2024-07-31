 ### general functions belonging to the 'multiFlux' module
#import matplotlib
#matplotlib.use('agg')

from constants import *
from emissionFn2 import *
from rateFn import * 
    

###python packages
#from numba import jit
import numpy as np
#import matplotlib.pyplot as plt
import os



def Npf(Ep):    
    return  N_p0 * (Ep / Ep0_j)**(-pp) * np.exp(- Ep / Epcut_j)

#@jit(nopython=True)
def fabs(Eg,taugg):
    factor = np.zeros_like(Eg) + 1
    
    for i in range(len(Eg)):
        if(taugg[i]>1e-4):
            factor[i] = (1- np.exp(- taugg[i] ) ) / taugg[i] 
  
    return factor


def NeSSC(Ee, Esyn_j, eproc):
    
    if os.path.exists(folder_nph + 'NeSSC_%d.txt'%eproc):    
        print("    Loading NeSSC_%d function ..."%eproc)
        Nx = np.loadtxt(folder_nph + 'NeSSC_%d.txt'%eproc)
        
        return Nx
    
    else:
        
        Pe_Loss_0 = Psyn_tot(Ee,Bj) + PIC_KN_tot(Ee,epsB,nph_0) + small
        hat_Ne_0 = N_LeakyBox(Ee,hat_Qe,Pe_Loss_0,Tau_ad) 
        hat_Ue_0 = Uphf(Ee,hat_Ne_0)
        Qe0_0 = etae * Pdiss / (G_jet**2 * beta_j*c * np.pi*Rrec**2 * hat_Ue_0 )
        Ne_0 = Qe0_0 * hat_Ne_0
        
        print("Esyn_j:",len(Esyn_j))
        
        jnuj_syn_0 =  h / (4*np.pi) * dP_dEdV_syn(Esyn_j,Ee,Ne_0,Psyn_1) 
        Inuj_syn_0 =  jnuj_syn_0 * Rrec
        nphj_syn_0 = 4 * np.pi * Inuj_syn_0 / (c*h* Esyn_j) 
        
        ## Ne, 1st order iteration
        lnph_1 = SumCurvLog( lepsB , np.log10(nph_0) , np.log10( Esyn_j / eV ) , np.log10( nphj_syn_0 ) )
        nph_1 = 10**lnph_1

        Pe_Loss_1 = Psyn_tot(Ee,Bj) + PIC_KN_tot(Ee,epsB,nph_1)  + small        
        hat_Ne_1 = N_LeakyBox(Ee,hat_Qe,Pe_Loss_1,Tau_ad) 
        hat_Ue_1 = Uphf(Ee,hat_Ne_1)
        Qe0_1 = etae * Pdiss / (G_jet**2*beta_j*c *np.pi*Rrec**2 * hat_Ue_1 )
        Ne_1 = Qe0_1 * hat_Ne_1
        
        np.savetxt(folder_nph + 'NeSSC_%d.txt'%eproc,Ne_1)
        
        return Ne_1
    
    
    
def AddComp(cbase,hlist,Ecomp,ELEcomp,name,write):
    '''
    This function appends the emission components as columns 
    of the same size, and write them into an output txt file.
    '''
    
    lcomp_w = SumCurvLog(lE, lELEB, np.log10(Ecomp / eV ),  np.log10(ELEcomp)  ) 
    comp_w = 10**lcomp_w    

    cbase.append(comp_w)
    cbase = np.array(cbase)
    hlist.append(name)

    if(write==1):
        fname = folder_dump + file_dump 
        header = "  ".join(hlist)
        fmt = '%1.6e'
        np.savetxt(fname, cbase.T,fmt=fmt,header=header)




def XsAddComp(cbase, header_list, ELEcomp, comp_name, out):
    '''
    This function appends the emission components (evaluated at
    XS energy points) as columns of the same size, and write them
    into an output txt file.
    '''
 
    cbase.append(ELEcomp)
    #header_list.append(comp_name)

    #if(out != 0):
        #arr_base = np.array(cbase)
        #fname = wfile  
        #header = "  ".join(header_list)
        #fmt = '%1.6e'
        #np.savetxt(fname, arr_base.T,fmt=fmt,header=header)




def WhereinXS(Emin, Emax,XS):
    Ein =[]
    idxs = []
    for i in range(len(XS)):
        if( (XS[i]>=Emin) and (XS[i]<=Emax) ):
            Ein.append(XS[i])
            idxs.append(i)
    
    return np.array( Ein ), idxs 


def intern_ntarget(Ee_1,Ne_1,
                   Bj,rb):

    Rrec = rb ## everywhere
    RT = 4 / 3 * Rrec
    #Rrec = 1e18
    
    ### Energies domains, source frame:
    ### primary electrons
    DlEph = 0.05
    
    ##Energy domain of electron-synchrotron photons
    Ece_m = 3/4/np.pi *qe *h *Bj / me /c * (Ee_1[0] / mec2)**2
    Ece_M = 3/4/np.pi *qe *h *Bj / me /c * (Ee_1[-1] / mec2)**2
    Esyn_1 = UniLogEGrid(1e-4*Ece_m, Ece_M,DlEph)
    
    
    
    ## Steady distribution of pirmary electrons, imposed as an stationary, broken, power-law:
    ## starting point: iteration 0

    
    jnu_syn_1 =  h / (4*np.pi) * dP_dEdV_synAha(Esyn_1,Ee_1,Ne_1,Bj,me) 
    Inu_syn_1 =  jnu_syn_1 * Rrec 
    nph_syn_1 = 4 * np.pi * Inu_syn_1 / (c*h* Esyn_1) 
    
    
    ###updating the total target photon field
    #lnphB = SumCurvLog( np.log10(epsB/eV) , lnphB, np.log10( Esyn_1 / eV ) , np.log10( nph_syn_1 ) )
    #nphB = 10**lnphB
    
    return Esyn_1, nph_syn_1
    
    
    
    

def comovSSCout(Ee_1,Ne_1,
               epsB,nphB,
               Bj,rb,
               XsEM,comps):

    '''
    This function returns the differential luminosity of leptonic and hadronic radiation
    escaping a magnetised, spherical volume.
    Setting comps == 1, the function returns a list of Nc numpy arrays, of size XsEM which corresponding to 
    each emission component. Nc is the number of emission components calculated within the body of this
    function, and the last array of the list is the total differential luminosity.
    Setting comps != 1, the function returns a single numpy array of size XsEM that contains
    the total differential luminosity escaping the sphere.
    
    '''


    EXs = XsEM
    
    Xs_cbase = []
    #XS_header_list.append('Xs_E[eV]')
    
    
    Tau_ad = rb / c
    Rrec = rb ## everywhere
    Vrec = 4 / 3 * np.pi * Rrec**3
    RT = 4 / 3 * Rrec
    #Rrec = 1e18
    
    ### Energies domains, source frame:
    
    ### primary electrons
    DlEe = 0.05
    DlEph = 0.05
    
    #Ee_1 = UniLogEGrid(Ee0,10*Eecut,DlEe)
    
    #epsB = UniLogEGrid(1e-5*eV, mec2,DlEph)
    #lnphB = np.zeros_like(epsB) + np.log10(small)
    
    
    ##Energy domain of electron-synchrotron photons
    Ece_m = 3/4/np.pi *qe *h *Bj / me /c * (Ee_1[0] / mec2)**2
    Ece_M = 3/4/np.pi *qe *h *Bj / me /c * (Ee_1[-1] / mec2)**2
    Esyn_1 = UniLogEGrid(1e-4*Ece_m, Ece_M, DlEph)
    
    ## Steady distribution of pirmary electrons, imposed as an stationary, broken, power-law:
    ## starting point: iteration 0
    #hn_e = (Ee_1 / Ee0)**(-pe) * ( 1 + (Ee_1 / Eeb)**(peb - pe ) )**(-1) * np.exp(- Ee_1 / Eecut)
    #Ne_1 = N_e0 * hn_e
    
    jnu_syn_1 =  h / (4*np.pi) * dP_dEdV_synAha(Esyn_1,Ee_1,Ne_1,Bj,me) 
    #Inu_syn_1 =  jnu_syn_1 * Rrec 
    #nph_syn_1 = 4 * np.pi * Inu_syn_1 / (c*h* Esyn_1) 
    
    ###updating the total target photon field
    #lnphB = SumCurvLog( np.log10(epsB/eV) , lnphB, np.log10( Esyn_1 / eV ) , np.log10( nph_syn_1 ) )
    #nphB = 10**lnphB
    UphB = Uphf(epsB,nphB) # photon field density (erg cm^-3)
    ELEsyn_1 = 4*np.pi*Rrec**3 / 3  * Esyn_1 * 4*np.pi* jnu_syn_1 / h 
    Xs_ELEsyn_1 =  Project_to_Xs(Esyn_1, ELEsyn_1, EXs)    
    Xs_cbase.append(Xs_ELEsyn_1)
  
    ##IC spectrum of primary e 
    #print("\n    Computing IC emission of primary electrons :")            
    # Energy domain of IC photons from primary e
    EIC_1 = UniLogEGrid(1e2 * eV, 10*TeV ,DlEph)
    jnu_IC_1 =  h / (4*np.pi) * EIC_1 * dN_dtdEdV2_IC(EIC_1,Ee_1,Ne_1, epsB,nphB)
    
    taugg_IC_1 = tau_gg( EIC_1, epsB, nphB, 4/3*Rrec ) 
    fabs_IC_1  =  fabs( EIC_1, taugg_IC_1 )
    #Inu_IC_1  =  jnu_IC_1 * Rrec * fabs_IC_1 
    #nph_IC_1  = 4 * np.pi * Inu_IC_1 / ( c * h * EIC_1) 
    
    ELEIC_1 = 4*np.pi*Rrec**3 / 3 * EIC_1 * 4*np.pi* jnu_IC_1 / h  * fabs_IC_1
    #print("    done.")

    Xs_ELEIC_1 =  Project_to_Xs(EIC_1, ELEIC_1, EXs)    
    Xs_cbase.append(Xs_ELEIC_1)
    
    if(comps==1):
        Xs_cbase.append(sum(Xs_cbase))
        return np.array(Xs_cbase)
    
    else:
        return sum(Xs_cbase)
    







