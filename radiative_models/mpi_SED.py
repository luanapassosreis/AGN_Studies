#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
Script Name: mpi_SED.py
Run Command: mpirun -n 4 python mpi_SED.py
Author: Luana Passos Reis
Date: 2025
==============================================================================

Description:
------------
This script computes the Spectral Energy Distribution (SED) for 
Synchrotron and Pion Decay emission processes. It utilizes MPI for 
parallel processing.

Outputs:
--------
- A plot of SEDs saved as 'SED_plot.png'.

Dependencies:
-------------
- numpy
- astropy
- naima
- mpi4py
- matplotlib

==============================================================================
"""

# ============================================================================
# =========== Import External Modules and Custom Functions ================== 
# ============================================================================


import matplotlib.pyplot as plt
from mpi4py import MPI

# ============================================================================
# ========================= MPI Initialization ===============================
# ============================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================================
# ========================= Generate and Save SED Plot =======================
# ============================================================================
if rank == 0:
    # Define the energy range for the SED
    spectrum_energy = np.logspace(2, 20, 1000) * u.eV  # 10^2 to 10^20 eV

    # Create the particle distribution
    particle_dist = create_particle_distribution()

    # Initialize emission models
    synch_model = SynchrotronSED(particle_dist)
    pion_model = PionDecaySED(particle_dist)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    synch_model.plot(ax, spectrum_energy)
    pion_model.plot(ax, spectrum_energy)

    # Plot settings
    ax.set_xlabel("Photon Energy [eV]")
    ax.set_ylabel("SED [erg / (cm² s eV)]")
    ax.set_title("SED: Synchrotron & Pion Decay")
    ax.legend()
    ax.grid(which="both", linestyle="--")

    # Save the figure
    plt.savefig("SED_plot.png", dpi=300)
    print("SED plot saved as 'SED_plot.png'.")

MPI.Finalize()
