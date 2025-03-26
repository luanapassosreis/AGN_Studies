from astro_constants import *
from source_info import *
from parameters import *

from scipy.integrate import quad
from scipy import special
import numpy as np
import math

'''
XL22: Xu & Lazarian
'''


# ============================================
# =========== Timescales [s] ================= 
# ============================================


def t_acc_fermi(va, delta_RX):
    '''ref: Eqs.(69) & (38) from XL22.
    timescale for Fermi particle acceleration (from reconnection)'''
    
    v_in = 0.05 * va
    beta_in = v_in / c
    d_ur = 2 * beta_in * ( 3 * beta_in**2 + 3*beta_in + 1 ) / ( 3 * (beta_in + 0.5) * (1 - beta_in**2) )
    
    return 4 * delta_RX / (c * d_ur)


def t_acc_drift(E, B, va):
    '''ref: Eq.(7) from M. V. del Valle, E. M. de Gouveia Dal Pino & G. Kowal 2017 e de Gouveia Dal Pino & Kowal 2015.
    timescale for Drift particle acceleration (from reconnection)'''
    
    v_rec = 0.05 * va
    
    return E / (qe*B*v_rec)


# ============================================
# ======= Photon Fields [erg-1 cm-3] ========= 
# ============================================


## OUV

eps_min = (12398/2500) * eV # [erg] 1 angstrom = 12398 eV
eps_max = (12398/1050) * eV # [erg] 
    
eps_OUV = np.logspace(np.log10(eps_min), np.log10(eps_max), num=100)

def n_OUV(eps):
    """Isotropic OUV photon field density [cm-3 erg-1]."""
    r_OUV = 100 * R_s  # OUV radius [cm]

    ## Energy density
    L_2keV = 7e43  # Luminosity at 2 keV [erg/s] (Lx)
    log_L_2500A = (1 / 0.760) * (np.log10(L_2keV) - 3.508)
    L_2500A = 10**log_L_2500A  # Luminosity at 2500 Angstroms [erg/s]
    U_OUV = L_2500A / (4 * np.pi * c * r_OUV**2)  # Energy density [erg/cm^3]

    dnOUVde = np.zeros_like(eps)  # Initialize the array to store the results
    
    dnOUVde = U_OUV * eps**(-2)

    return dnOUVde

nph_OUV = n_OUV(eps_OUV) # [cm^{-3} erg^{-1}]




## X-Rays

eps_x = np.logspace(np.log10(2 * keV), np.log10(200 * keV), num=100)  # [eV to MeV]

def n_Xrays(eps):
    """Isotropic photon field density [cm-3 erg-1].
    Eq. (4) from Mbarek et al. 2023."""
    r_c = 10 * R_s  # [cm] coronal size
    L_x = 7 * 10**43  # [erg s^-1]
    epsilon_0 = 7 * keV  # [erg] (compton hump energy)
    
    U_x = L_x / (4 * np.pi * c * r_c**2)  # [erg cm^-3]
    
    dnxde = np.zeros_like(eps)  # Initialize the array to store the results
    
    for i in range(len(eps)):
        if eps[i] < (20 * keV):
            dnxde[i] = U_x * epsilon_0**(-2)
        elif (20 * keV) <= eps[i] <= (200 * keV):
            dnxde[i] = U_x * eps[i]**(-2)
        else:
            0.
    
    return dnxde

nph_x = n_Xrays(eps_x)  # [cm^{-3} erg^{-1}]



# ============================================
# ======= Rates [s-1] = 1 / timescale ======== 
# ============================================


def rate_bremss_e(n, Z, Ee):
    '''Eq. (17) of Romero et al. (2010a)'''
    
    return 4 * n * Z**2 * re**2 * alpha_fine * c * (np.log(2*Ee / (mec2)) - 1/3)


def rate_synch(E, B, m):
    '''Eq. (5) of Romero et al. (2010)'''
    
    UB = B**2 / (8*np.pi)
    
    return 4/3 * sigmaT * c * (UB/mec2) * (me/m)**3 * E/(m * c**2)
    

def rate_IC(Ee, eps, nph):
    
    def integral_right(Ee, eps, nph):
        gamma = 4 * eps * Ee / mec2**2
        emin = eps
        emax = (gamma * Ee) / (1 + gamma)
        dlog_e = 0.01  # logarithmic step size in energy
        log_e = np.arange(np.log10(emin/eV), np.log10(emax/eV), dlog_e) #log(E/eV) photon energy array in log
        energies = 10**log_e * eV  # photon energy array in linear energy units (eV)
        de = np.diff(energies)  # small energy differences
        
        # if emax <= emin:
        #     print(f"Invalid energy range: emin={emin}, emax={emax}")

        cte = (2 * np.pi * re**2 * me**2 * c**5) / Ee**2

        def f(q):
            return 2*q * np.log(q) + (1+2*q)*(1-q) + (1-q)*(gamma*q)**2 / (2*(1+gamma*q))

        Sum = 0
        for i in range(np.size(de)):
            q = energies[i] / (gamma * (Ee - energies[i]))
            Sum = Sum + de[i]*(energies[i]-eps) *cte * (nph/eps) * f(q)
        
        return Sum
        
    deps = np.diff(eps)
    
    SSum = np.zeros_like(Ee)
    
    for j in range(np.size(SSum)):
        for i in range(np.size(deps)):
            SSum[j] = SSum[j] + ( deps[i] * integral_right(Ee[j], eps[i], nph[i]) )
            
    rate = (1 / Ee) * SSum
    
    
    if np.any(nph <= 0):
        print("Warning: Some photon densities are zero or negative")

    
    return rate


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



#### maybe wrong


# Define cross-section function for pγ interactions
def sigma_pg(eps_prime):
    """ Cross-section function for pγ interactions. """
    sigma1 = 340 * 1e-6 * barn  # Peak cross-section
    K1 = 0.2  
    sigma2 = 120 * 1e-6 * barn
    K2 = 0.6  

    if eps_prime < 200 * MeV:
        return 0.
    elif 200 * MeV <= eps_prime < 500 * MeV:
        return sigma1 * K1 * 0.5 * (eps_prime**2 - (200 * MeV)**2)
    else:
        return (sigma1 * K1 * 0.5 * ((500 * MeV)**2 - (200 * MeV)**2) +
                sigma2 * K2 * 0.5 * (eps_prime**2 - (500 * MeV)**2))

# Function to integrate over the cross-section
def integrate_sigma(eps_prime_min, eps_prime_max):
    """ Integrates eps' * sigma_pg(eps') from eps_prime_min to eps_prime_max """
    return quad(lambda eps_prime: eps_prime * sigma_pg(eps_prime), eps_prime_min, eps_prime_max, limit=100)[0]

# Function to compute the inverse bremsstrahlung cooling time
def rate_bth_cool(Ep, eps, nph):
    """
    Computes the inverse bremsstrahlung cooling rate.

    Parameters:
    Ep : array-like
        Proton energy in erg.
    eps : array-like
        Photon energy array in erg.
    nph : array-like
        Photon number density per energy bin.

    Returns:
    Array with cooling rate values.
    """
    
    eps_th = 2 * MeV
    
    gp = Ep / mpc2  # Proton Lorentz factor
    deps = np.diff(eps)
    dleps = np.log10(eps[1] / eps[0])

    Ieps = np.zeros_like(gp) + 1e-30  # Avoid division by zero

    for i in range(len(gp)):
        eps_inf = eps_th / (2 * gp[i])
        
        if eps_inf >= eps[0] and eps_inf < eps[-1]:
            j0 = int(math.ceil(np.log10(eps_inf / eps[0]) / dleps))
            for j in range(j0, len(deps)):
                eps_prime_min = eps_th
                eps_prime_max = 2 * gp[i] * eps[j]
                
                # Compute the integral of eps' * sigma_pg(eps') over the given range
                integral_sigma = integrate_sigma(eps_prime_min, eps_prime_max)

                Ieps[i] += deps[j] * nph[j] / (eps[j]**2) * integral_sigma

    return c / (gp**2) * Ieps







# ============================================
# ======= Power Emitted [?]  ======== 
# ============================================


# def P_synch():    
    # return 4/3 * sigmaT * c * UB * (Ee/mec2)**2