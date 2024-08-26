from constants import *  ## cgs units
from rateFn import *
from comovELEs import *
import os 


def modelSSC(Cs, Xs, comps):
    '''
    This function returns the emission curves from the SSC model parameters
    '''
    Le      = 10**Cs[0]         #part of Pdiss to accelerate electrons
    E0      = 10**Cs[1]*eV      #electron distribution normalization characteristic energy
    Ecut    = 10**Cs[2]*eV      #electron distribution maximum energy
    p       = Cs[3]             #electron distribution power law index
    pb      = Cs[4]             #electron distribution power law index
    Bj      = 10**Cs[5]         #magnetic field in the jet frame
    f_r     = Cs[6]             #fraction of the maximum radius of the emitting region given by causality
    G_jet   = Cs[7]             #bulk lorentz factor of the emiting blob

    rb    = f_r * c * At_v * G_jet / (1+zr) ### emitting blob radius

    tEsyn_pk = (1+zr)/G_jet * Esyn_pk
    Eb = EeSynch(tEsyn_pk,Bj)


    ##Transforming the data photon energy to the jet co-moving frame    
    Xs = Xs * (1. + zr ) / G_jet
   
    DlE = 0.05
    E_1 = UniLogEGrid(E0,100*Ecut, DlE)
    hn_e = (E_1 / E0)**(-p) / ( 1 + (E_1 / Eb) )**(pb - p )  * np.exp(- E_1 / Ecut)
    Ie = Uphf(E_1, hn_e)
    N_e0 = Le / (G_jet**2 * c *np.pi * rb**2 * Ie )
    Ne_1 = N_e0 * hn_e
    eps, nph = intern_ntarget(E_1,Ne_1,Bj,rb)    
    

    ### Calculate the synch and SSC differential luminosity, as seen in the jet co-moving frame.
    comovEM_ELE = comovSSCout(E_1,Ne_1,eps,nph,Bj,rb,Xs, comps)
    return comovEM_ELE * G_jet**4 * (4*np.pi*DL**2)**(-1)


def chi2(Cs):
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

### Observed EM feautures (from the SED)
At_v        =  60 * 60 * 2  ### minumum variability time interval  (2 hrs)
Esyn_pk     = 2e2*eV


### Observational EM data
obsDataFiles = ['EMobs.txt', 'EMobs_up.txt']        
EMdata = np.loadtxt(obsDataFiles[0], unpack = 1)
EMdata_up = np.loadtxt(obsDataFiles[1], unpack = 1)

Xs_EMea = 10**EMdata[0]* eV
Ys_EMea = 10**EMdata[1] # erg cm^2 s^-1
sigmas_EMea = 10**EMdata_up[1] - 10**EMdata[1] 

## Give more weight for low energy data points
weighted_s=0
if(weighted_s==True):
    w0=0.05
    E_sigma = 1e8*eV
    w_sigma = w0 + np.exp(-Xs_EMea/E_sigma)
    sigmas_EMea =sigmas_EMea*w_sigma

### parameters limits
C_lims = []

Le_min, Le_max = 1e43 , 6e44  
C_lims.append([ np.log10(Le_min) , np.log10(Le_max)  ])
OutFileHeader = "C0=log10(Le), "

E0_min, E0_max = 5e2*mec2 , 5e3*mec2 
C_lims.append([ np.log10(E0_min/eV) , np.log10(E0_max/eV) ])
OutFileHeader += "C1=log10(E0/eV), "

Ecut_min, Ecut_max = 6e4 *mec2 , 1e6 *mec2 
C_lims.append([ np.log10(Ecut_min/eV) , np.log10(Ecut_max/eV) ])
OutFileHeader += "C2=log10(Ecut/eV), "

p_min, p_max = 1.5 , 2.1
C_lims.append([p_min ,  p_max ])
OutFileHeader += "C3 = p, "

pb_min, pb_max = 3. , 4.5
C_lims.append([pb_min ,  pb_max ])
OutFileHeader += "C4 = pb, "

Bj_min, Bj_max = 1e-3, 1e-1 
C_lims.append([ np.log10(Bj_min) , np.log10(Bj_max) ])
OutFileHeader += "C5 = np.log10(Bj), "

f_r_min, f_r_max = 0.2, 1.
C_lims.append( [ f_r_min , f_r_max ] )
OutFileHeader += "C6 = f_r, "

G_jet_min, G_jet_max = 20 , 45
C_lims.append( [G_jet_min , G_jet_max ] )
OutFileHeader += "C7 = G_jet"


C_lims = np.asarray(C_lims)
# Definig the range of the parameters
Cm = C_lims[:,0]
CM = C_lims[:,1]
N_data = len(Xs_EMea)
Nc = len(C_lims)
