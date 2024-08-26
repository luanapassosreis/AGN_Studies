#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Name: mpi_AGA.py
Run: mpirun -n 4 python mpi_AGA.py
Author: 
Date: 
=============================================================================
=============================================================================
Description: 
Inputs:
Outputs: 

Comments: All internal calculations are done in cgs units
==============================================================================
"""
# ============================================================================
# =========== Imports external files containing useful functions ============= 
# ============================================================================
from MetaHeuristics import *
from Chi2Set import *
#from striped_Chi2Set import *
from outsFn import *
from constants import *  ## cgs units
# ============================================================================
# ======================= Imports python libraries =========================== 
# ============================================================================
import os
import sys
import warnings
import random
import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
from math import ceil, exp
import matplotlib.pyplot as plt
from random import choice, shuffle
from random import randint, uniform

# ============================================================================
# =========== Fixed parameters and initialization of parameters ==============
# ============================================================================
ObFunc = chi2         ## Objective function (striped_chi2 or chi_2)
N0 = 25               ## number of inital parents, N0 = 64 
it_max = 100          ## maximum number of iterations, it_max = 1_000
runs = 1              ## statistical analysis, runs = 120 

runs_convs=runs
stepout = 1
chis = np.zeros(runs)
histogs=[]
labels=[]
convs=[]
label= "AGA"
parallel = False
# ============================================================================
# ======================= Initialize virtualization ========================== 
# ============================================================================
if parallel:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()                           # number of the rank, starting from 0
	n_ranks = comm.Get_size()                        # number of ranks used, nodes
else:
	rank = 0  
	n_ranks = 1
	
runs_rank = int(runs/n_ranks)                        # number of tasks per node
if rank == 0:
    runs_rank = runs - runs_rank*(n_ranks-1)
print(runs_rank ," runs of " + label + " in rank ", rank)

itsols = np.zeros((it_max+1,runs_rank))
param  = np.zeros((len(Cm),runs_rank))
time   = np.zeros(runs_rank)

# ============================================================================
# ============================ Optimization cycle ============================ 
# ============================================================================
for run in range(runs_rank):
    alh = aga(N0, int(np.sqrt(N0)), ObFunc, Cm, CM, Nc, it_max)

    if( run%stepout == 0):
        print("===>>>"+label+" run %d"%run)
    itsols[:,run]  = alh.get_zhistory()
    param[:,run]   = alh.get_Gbest()
    time[run]      = alh.get_time()

# ============================================================================
# ====================== Directory to save results =========================== 
# ============================================================================
dir_save = "Outs"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
dir_save += "/"

dir_merged = "MergedOuts"
if not os.path.exists(dir_merged):
    os.makedirs(dir_merged)
dir_merged += "/"

# ============================================================================
# ============================ Saving results ================================ 
# ============================================================================ 
filename_conv   = dir_save + "mpi_" + label + "_convsChiS_"  + str(rank) + "_ITS.npy"    
filename_sols   = dir_save + "mpi_" + label + "_sols_"       + str(rank) + "_ITS.npy"    
filename_time   = dir_save + "mpi_" + label + "_time_"       + str(rank) + "_ITS.npy"    
np.save(filename_conv,itsols)
np.save(filename_sols, param)
np.save(filename_time,  time)

if parallel:
	MPI.Finalize()
if rank == 0:
        concat(label,n_ranks,dir_save, dir_merged)