## Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy import units as u
from astropy import constants as const
from scipy.stats import chisquare
from scipy.optimize import curve_fit





# c = (const.c).to('cm s-1')
# m_e = (const.m_e).to('g')
# m_p = (const.m_p).to('g')
# sigma_T = (const.sigma_T).to('cm2')
# e = (const.e).value / (3.33564e-10) * u.statcoulomb

# alpha_f = 1/137 # fine structure constant [dim.]
# r_0 = 2.8179e-13 * u.cm # electron classical radius [cm]



## Source parameters

erg = 1.602177e-12 # Convert eV to erg

m = 2e7 # m = M / M_sun: BH mass in solar units
mdot = 0.7 # Mdot / Mdot_Edd - radiation dominated regime

l = 10    # L / R_s
l_x = 5   # L_x / R_s - testar = 1 p simplificação inicial e depois vemos no espaço paramétrico
r_x = 6   # R_x / R_s




## Fixed Constants

Mdot_Edd = 1.45e18 * m # [g s-1] (KGS15 page 4)
R_s = 2.96e5 * m # [cm] = 2GM/c^2 (KGS15 page 4)

c = 2.9979e10 # [cm/s]
m_e = 9.1093e-28 # [g]
m_p = 1.6726e-24 # [g]
sigma_T = 6.6524e-25 # [cm2]
e = 4.8032e-10 # [statcoul]

alpha_f = 1/137 # [dim.] fine structure constant
r_0 = 2.8179e-13 # [cm] electron classical radius

# q = ( 1 - ( 3 * R_s / R_X)**(1/2) )**(1/4) # KGS15 page 4
# Gamma = (1 + (v_A0 / c )**2 )**(-1/2) # KGS15 page 4

q = ( 1 - ( 3 / r_x )**(1/2) )**(1/4)
va0 = c # [cm/s] Alfvén Speed (Khiali et al. 2015 page 38)
Gamma = 1 / np.sqrt(2)




# def mag_rec_power(Gamma, r_X, l, l_X, q, mdot, m):
#     ''' Magnetic recconection power released by turbulent fast reconnection in the surrounds of the BH.
#     Eq. (15) of Kadowaki, de Gouveia Dal Pino & Singh 2015 (KGS15).'''
#     # [erg s-1]
    
#     return 1.66e35 * Gamma**(-1/2) * r_X**(-5/8) * l**(-1/4) * l_X * q**(-2) * mdot**(3/4) * m

# def coronal_mag_field(r_x, mdot, m):
#     '''Inner disk magnetic field intensity.
#     Eq.(2) of KGS15.'''
#     # [G]
    
#     return 9.96e8 * r_x**(-5/4) * mdot**(1/2) * m**(-1/2)

# def coronal_density(Gamma, r_x, l,  q, mdot, m):
#     '''Eq.(7) of KGS15.'''
#     # [cm-3]
    
#     return 8.02e18 * Gamma**(1/2) * r_x**(-3/8) * l**(-3/4) * q**(-2) * mdot**(1/4) * m**(-1)

# def coronal_temperature(Gamma, r_x, l, q, mdot):
#     '''Eq.(6) of KGS15.'''
#     # [K]
    
#     return 2.73e9 * Gamma**(1/4) * r_x**(-3/16) * l**(1/8) * q**(-1) * mdot**(1/8)

# def width_current_sheet(Gamma, r_x, l, l_x, q, mdot, m):
#     '''Eq.(14) from ERRATUM of KGS15.'''
#     # [cm]
    
#     return 11.6 * Gamma**(-5/4) * r_x**(31/16) * l**(-5/8) * l_x * q**(-3) * mdot**(-5/8) * m



class BHPhysics:
    def __init__(self, Gamma, r_x, l, l_x, q, mdot, m):
        self.Gamma = Gamma
        self.r_x = r_x
        self.l = l
        self.l_x = l_x
        self.q = q
        self.mdot = mdot
        self.m = m
    
    def mag_rec_power(self):
        ''' Magnetic recconection power released by turbulent fast reconnection in the surrounds of the BH.
        Eq. (15) of Kadowaki, de Gouveia Dal Pino & Singh 2015 (KGS15).'''
        # [erg s-1]
        return 1.66e35 * self.Gamma**(-1/2) * self.r_x**(-5/8) * self.l**(-1/4) * self.l_x * self.q**(-2) * self.mdot**(3/4) * self.m

    def coronal_mag_field(self):
        '''Inner disk magnetic field intensity.
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