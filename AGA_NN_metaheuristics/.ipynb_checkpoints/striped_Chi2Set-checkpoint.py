'''
Chi^2 objective function for a SSC one zone leptonic model
based on reconnection acceleration in a stripd jet
'''

from constants import *  ## cgs units
from striped_jet import StripJetPas
from rateFn import *
from comovELEs import *
import os 

def modelSSC(Cs, Xs, comps):
    '''
    This function returns the emission curves from the SSC model parameters
    '''
    f_r     = Cs[0]             #fraction of the maximum radius of the emitting region given by causality
    E0      = 10**Cs[1]*eV      #electron distribution normalization characteristic energy
    Ecut    = 10**Cs[2]*eV      #electron distribution maximum energy
    p       = Cs[3]             #electron distribution power law index
    pb      = Cs[4]             #electron distribution power law index
    eta     = 10**Cs[5]         #fraction of Pdiss to accelerate electrons
    a       = Cs[6]             #stripe distribution power spectral index
    f_lmin  = 10**Cs[7]         #minimal stripe width in terms of Rg
    G8      = Cs[8]             #terminal lorenz factor
    z       = Cs[9]             #position in the jet

    ####****** BACKGROUND JET PARAMETRISATION **********
    pas_co = StripJetPas(z, G8, lamb, a=a, f_lmin= f_lmin, MBH=MBH)
    Bj    = pas_co[0] 
    G_jet = pas_co[1]
    Pdiss = pas_co[2]
    rb    = f_r * c * At_v * G_jet / (1+zr) ### emitting blob radius
    ####*************************************************

    #### ********** EMITTING PARTICLES PARAMETRISATION ******
#   tEsyn_0     =  (1+zr)/G_jet * Esyn_0 
    tEsyn_pk    =  (1+zr)/G_jet * Esyn_pk 
#   tEsyn_cut   =  (1+zr)/G_jet * Esyn_cut 

    Le = eta * Pdiss
#   E0     = EeSynch(tEsyn_0,Bj)
    Eb     = EeSynch(tEsyn_pk,Bj)
#   E_ecut  = EeSynch(tEsyn_cut,Bj)
    ####*******************************************************

    ##Transforming the data photon energy to the jet co-moving frame    
    Xs = Xs * (1. + zr ) / G_jet
   
    DlE = 0.05
    E_1 = UniLogEGrid(E0,100*Ecut, DlE)
    hn_e = (E_1 / E0)**(-p) / ( 1 + (E_1 / Eb) )**(pb - p )  * np.exp(- E_1 / Ecut)
    Ie = Uphf(E_1, hn_e)
    N_e0 = Le / (G_jet**2 * c *np.pi * rb**2 * Ie )
    Ne_1 = N_e0 * hn_e
    eps, nph = intern_ntarget(E_1,Ne_1,Bj,rb)    
    
    """
    print("Striped jet parameters:")
    print("MBH = %1.3e Msun"%(MBH/Msun))
    print("Radiating blob parameters:")
    print("B_diss = %1.3e G  G_diss = %.3f  Pdiss = %1.3e erg/s"%(pas_co[0],pas_co[1], pas_co[2]) )
    print("z_pk = %1.3e cm, z_diss = %1.3e pc, rb_diss = %1.3e cm"%(pas_co[3],pas_co[4]/pc, rb) )
    print("sigmaBK = %.3f"%(pas_co[6]) )

    #imname1 = 'MBH_%1.1e/strip_fmin_%d_G8_%d_zd_%1.1f.png'%(MBH/Msun,f_lmin,G8,pas_co[4]/pc)

    print("XXXXXX Ee0 = %1.3e"%(E0/mec2))
    print("XXXXXX Eeb = %1.3e"%(Eb/mec2))
    print("XXXXXX Eecut = %1.3e"%(Ecut/mec2))

    print("Le=%1.3e"%(Le) )
    """


    ### Calculate the synch and SSC differential luminosity, as seen in the jet co-moving frame.
    comovEM_ELE = comovSSCout(E_1,Ne_1,eps,nph,Bj,rb,Xs, comps)
    return comovEM_ELE * G_jet**4 * (4*np.pi*DL**2)**(-1)


def striped_chi2(Cs):
    '''
    This function returns the reduced chi2 value
    '''    
    Xs_nuFnu = modelSSC(Cs, Xs_EMea, comps=0)
    Chi2 = sum( (Ys_EMea - Xs_nuFnu)**2 / sigmas_EMea**2 )
    Chi2_red = Chi2 / (N_data - Nc - 1)    
    return Chi2_red



#### SOURCE DATA 
zr      = 0.031         ###redshift
DL      = 122 * 1e6*pc  ###distance
MBH     = 2e8*Msun
####********************************************

### Observed EM feautures
At_v        =  60 * 60 * 2  ### minumum variability time interval  (2 hrs)
#Esyn_0     =  1.3e-1*eV  ### observed synchrotron lowest energy
Esyn_pk     =  2e2*eV     ### observed synchrotron peak      
#Esyn_cut   =  1.0e2*eV   ### observed synchrotron cut-off
####********************************************

### Observational EM data
obsDataFiles = ['EMobs.txt', 'EMobs_up.txt']        
EMdata = np.loadtxt(obsDataFiles[0], unpack = 1)
EMdata_up = np.loadtxt(obsDataFiles[1], unpack = 1)

Xs_EMea = 10**EMdata[0]* eV
Ys_EMea = 10**EMdata[1] # erg cm^2 s^-1
sigmas_EMea = 10**EMdata_up[1] - 10**EMdata[1] 

weighted_s=0
if(weighted_s==True):
    w0=0.05
    E_sigma = 1e8*eV
    w_sigma = w0 + np.exp(-Xs_EMea/E_sigma)
    sigmas_EMea =sigmas_EMea*w_sigma

N_data = len(Xs_EMea)
####************************************

### fixed parameters
lamb = 80
#a       = 4
#f_lmin  = 1000
#G8      = 30
#z    = 5
#f_r = 10**(-0.4)

### parameters limits
C_lims = []

f_r_min, f_r_max = 0.2, 1.
C_lims.append( [ f_r_min , f_r_max ] )
#OutFileHeader += "C0 = f_r, "

E0_min, E0_max = 5e2*mec2 , 5e3*mec2 
C_lims.append([ np.log10(E0_min/eV) , np.log10(E0_max/eV) ])
#OutFileHeader += "C1 = log10(E0/eV), "

Ecut_min, Ecut_max = 6e4*mec2 , 1e6*mec2 
C_lims.append([ np.log10(Ecut_min/eV) , np.log10(Ecut_max/eV) ])
#OutFileHeader += "C2 = log10(Ecut/eV), "

p_min, p_max = 1.5 , 2.1
C_lims.append([p_min ,  p_max ])
#OutFileHeader += "C3 = p, "

pb_min, pb_max = 3. , 4.5
C_lims.append([pb_min ,  pb_max ])
#OutFileHeader += "C4 = pb, "

eta_min, eta_max = 1e-4, 1. 
C_lims.append([ np.log10(eta_min) , np.log10(eta_max) ])
#OutFileHeader += "C5 = np.log10(eta), "

a_min, a_max = 3.01 , 4.5
C_lims.append([a_min ,  a_max ])
#OutFileHeader += "C6 = a, "

f_lmin_min, f_lmin_max = 9e1, 1.1e3
C_lims.append([np.log10(f_lmin_min) ,  np.log10(f_lmin_max) ])
#OutFileHeader += "C7 = np.log10(f_lmin), "

G8_min, G8_max = 20. , 40.
C_lims.append( [G8_min , G8_max ] )
#OutFileHeader += "C8 = G8, "

z_min, z_max = 3. , 17.
C_lims.append([z_min ,  z_max ])
#OutFileHeader += "C9 = z "

C_lims = np.asarray(C_lims)
# Definig the range of the parameters
Cm = C_lims[:,0]
CM = C_lims[:,1]
Nc = len(C_lims)
####*****************************
