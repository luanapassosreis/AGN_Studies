from constants import *  ## cgs units
from Chi2Set import *
from PlotChi2 import *

###python packages
import numpy as np
import matplotlib.pyplot as plt

import os


def PlotFluxSED(curves, imname ):
    '''
    comment the parameters that are being imported as inputs
    '''

    font = {'family' : 'serif', 'weight' : 'normal', 'size' : 24 }
    plt.rc('font', **font)
         
    ### observed data points
    
    ### configuration of the plotting box.   
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 10) )#, gridspec_kw={'height_ratios': [1, 1.5]})
    ax1.set_xlim(1e-2, 1e14)
    ax1.set_ylim(5e-15, 1e-8)

    ## plotting the observed data points
    ax1.errorbar(Xs_EMea/eV, Ys_EMea, yerr = [sigmas_EMea, sigmas_EMea],fmt='s', markersize='8', color = 'red',
                capsize=5)
    ### plotting the curve models.
    ax1.plot( curves[clab[0]] , curves[clab[1]], '-',color = 'gray', linewidth=2,label = clab[1])
    ax1.plot( curves[clab[0]] , curves[clab[2]], '--', dashes=(8,6),color = 'gray', linewidth=2,label = clab[2])
    ax1.plot( curves[clab[0]] , curves[clab[3]], '-', color = 'c', linewidth=2,label = clab[3])

    #plot format
    plt.yscale('log')
    plt.xscale('log')
    ax1.set_ylabel(r"$\nu F_{\nu}$ [erg cm$^{-2}$ s$^{-1}$]", fontsize=28)
    ax1.set_xlabel(r"$E$ [eV]", fontsize=28)
    #ax1.set_title(r"$\chi^2$ = %.3f"%(Chi2) )
    #list_Mxticks = np.logspace(-6,20,14)
    #list_mxticks = np.logspace(-6,20,27)
    #ax1.set_xticks(list_Mxticks)  
    #ax1.set_xticks(list_mxticks, minor = True)  

    plt.tick_params( axis='both', labelsize=28)
    plt.tick_params( which='major', direction = 'in', length=18, width=1.2, colors='k', left=1, right=1, top =1)
    ax1.tick_params( which='minor', direction = 'in', length=10, width=1.2, colors='k',  left=1, labelbottom=False)#, right=1)
    
    #fig.suptitle('Assexual Genetic Algorithm (AGA)', fontsize=30)
    fig.suptitle('Differential Evolution (DE)', fontsize=30)
    ax1.legend(loc = 3, ncol=1 ,fontsize=20)
    plt.tight_layout()
    plt.savefig(imname)
    plt.close(fig)
    print("Plot saved in: ", imname)

dlE = 0.1
Xs_EM = UniLogEGrid(1e-3*Xs_EMea[0], 1e1*Xs_EMea[-1] , dlE)
clab = ["E_eV","eSyn","SSC","Total"]

#label = "AGA"
label = "DE"
run = best_run
out = np.load("MergedOuts/mpi_"+label+"_sols_ITS.npy")        
Cs = out[:,run]
curves = {clab[i+1] : value for i, value in enumerate(modelSSC(Cs, Xs_EM, comps=1))}
curves[clab[0]] = Xs_EM / eV

imname = "plots/"+label+"SED.png"
if not os.path.exists("plots"):
    os.makedirs("plots")

PlotFluxSED(curves, imname) 
