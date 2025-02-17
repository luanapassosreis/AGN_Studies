from astro_constants import *
from source_info import *
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


# ==============================================
# =========== Emission Region ================== 
# ==============================================

coronal_data = Coronal_Description(r_x, l, l_x, mdot, m)

B_c = coronal_data.coronal_mag_field()
n_c = coronal_data.coronal_density()
T_c = coronal_data.coronal_temperature()
delta_RX = coronal_data.width_current_sheet()
wdot_B = coronal_data.mag_rec_power()

output_filename = "coronal_parameters.txt"

with open(output_filename, "w") as file:
    file.write(f'Coronal Magnetic Field:\n B_c = {B_c:.4E} G\n\n')
    file.write(f'Coronal Particle Number Density:\n n_c = {n_c:.4E} cm-3\n\n')
    file.write(f'Coronal Temperature:\n T_c = {T_c:.4E} K\n\n')
    file.write(f'Width of the current sheet:\n delta_RX = {delta_RX:.4E} cm\n\n')
    file.write(f'Reconnection Power:\n wdot_B = {wdot_B:.4E} erg s-1\n')

print(f"Output saved to {output_filename}")



# ============================================================================
# ========================= ... =====================
# ============================================================================



rho = n_c * mp # fluid density [g cm-3]

v_a0 = B_c / np.sqrt(4 * np.pi * rho) # Alfvén speed
va = v_a0 * coronal_data.Gamma


with open(output_filename, "a") as file:  # "a" mode appends to the file
    file.write(f'\nNew entry:\n')
    file.write(f'Rho:\n rho = {rho:.4E} g cm-3\n\n')
    file.write(f'v_a0:\n v_a0 = {v_a0:.4E} cm s-1\n\n')
    file.write(f'va:\n va = {va:.4E} cm s-1\n\n')
    file.write(f'va/c:\n va/c = {va/c:.4f} \n\n')
    file.write(f'v_a0/c:\n v_a0/c = {v_a0/c:.4f}\n')

print(f"Output appended to {output_filename}")


# ============================================================================
# ========================= Define Particle Distribution =====================
# ============================================================================






# def create_particle_distribution():
#     """Create a power-law particle distribution."""
#     amplitude = 1e36 * u.Unit("1/eV")
#     e_0 = 1 * u.TeV  # Reference energy
#     alpha = -2  # Power-law index
#     return PowerLaw(amplitude, e_0, alpha)