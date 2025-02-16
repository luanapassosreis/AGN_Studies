from astro_constants import *
import numpy as np


'''
cgs units or normalized regarding M_sun, R_s & Mdot_Edd
'''

# ============================================
# =========== Energy [ergs] ================== 
# ============================================

Ep = np.logspace(6, 20, num=50) * eV  # [2,20]
Ee = np.logspace(2, 15, num=50) * eV  # [2,15]

Ep_drift = np.logspace(17.72, 20, num=50) * eV  # 5.23e+17 = 10^17.72
Ee_drift = np.logspace(14.46, 20, num=50) * eV  # 2.85e+14 = 10^14.46



# =========================================
# =========== the source ================== 
# =========================================

m = 2e7 # m = M / M_sun: BH mass in solar units
mdot = 0.7 # Mdot / Mdot_Edd - radiation dominated regime

l = 10    # L / R_s
l_x = 10   # L_x / R_s - testar = 1 p simplificação inicial e depois vemos no espaço paramétrico
r_x = 10   # R_x / R_s

va0 = c # [cm/s] Alfvén Speed (Khiali et al. 2015 page 38)
# va0 = B_c / np.sqrt(4 * np.pi * rho)

Mdot_Edd = 1.45e18 * m # [g s-1] (KGS15 page 4)
R_s = 2.96e5 * m # [cm] = 2GM/c^2 (KGS15 page 4)

# Gamma = (1 + (va0 / c )**2 )**(-1/2) # KGS15 page 4
# q = ( 1 - ( 3 * R_s / R_x)**(1/2) )**(1/4) # KGS15 page 4


# ============================================================================
# ========================= Define Particle Distribution =====================
# ============================================================================


def create_particle_distribution():
    """Create a power-law particle distribution."""
    amplitude = 1e36 * u.Unit("1/eV")
    e_0 = 1 * u.TeV  # Reference energy
    alpha = -2  # Power-law index
    return PowerLaw(amplitude, e_0, alpha)






class Coronal_Description:
    def __init__(self, r_x, l, l_x, mdot, m):
        self.r_x = r_x
        self.l = l
        self.l_x = l_x
        self.mdot = mdot
        self.m = m
        self.Gamma = 1 / np.sqrt(2)
        # self.Gamma = (1 + (va0 / c )**2 )**(-1/2)
        self.q = ( 1 - ( 3 / self.r_x )**(1/2) )**(1/4)

        
    def coronal_mag_field(self):
        '''Calculate inner disk magnetic field intensity.
        Eq.(2) of KGS15.'''
        # [G]
        return 9.96e8 * self.r_x**(-5/4) * self.mdot**(1/2) * self.m**(-1/2)

    
    def coronal_density(self):
        '''Calculate coronal particle number density.
        Eq.(7) of KGS15.'''
         # [cm-3]
        return 8.02e18 * self.Gamma**(1/2) * self.r_x**(-3/8) * self.l**(-3/4) * self.q**(-2) * self.mdot**(1/4) * self.m**(-1)

    
    def coronal_temperature(self):
        '''Calculate coronal temperature.
        Eq.(6) of KGS15.'''
        # [K]
        return 2.73e9 * self.Gamma**(1/4) * self.r_x**(-3/16) * self.l**(1/8) * self.q**(-1) * self.mdot**(1/8)
    
    
    def width_current_sheet(self):
        '''Calculate the width of the current sheet.
        Eq.(14) from ERRATUM of KGS15.'''
        # [cm]
        return 11.6 * self.Gamma**(-5/4) * self.r_x**(31/16) * self.l**(-5/8) * self.l_x * self.q**(-3) * self.mdot**(-5/8) * self.m
    
    
    def mag_rec_power(self):
        ''' Magnetic recconection power released by turbulent fast reconnection in the surrounds of the BH.
        Eq. (15) of Kadowaki, de Gouveia Dal Pino & Singh 2015 (KGS15).'''
        # [erg s-1]
        return 1.66e35 * self.Gamma**(-1/2) * self.r_x**(-5/8) * self.l**(-1/4) * self.l_x * self.q**(-2) * self.mdot**(3/4) * self.m