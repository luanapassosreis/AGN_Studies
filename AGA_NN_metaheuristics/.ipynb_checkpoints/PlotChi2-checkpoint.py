'''
plot of the evolution of Chi^2 as a function of iteration
'''
import numpy as np
import matplotlib.pyplot as plt
import os


def PlotChi2():
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8) )
    ax1.set_xlim(1, 1e3)
    #ax1.set_ylim(1e-1, 1e3)

    #average
    ax1.plot( it, np.mean(outMat, axis = 1) , color ='c', linewidth=3, label = label)
    
    #best
    ax1.plot( it, outMat[:,best_run],'--', color ='c', linewidth=3)

    plt.yscale('log')
    plt.xscale('log')
    ax1.set_ylabel(r"$\chi ^2 _{red}$", fontsize=28)
    ax1.set_xlabel(r"Iteration", fontsize=28)
    ax1.legend(loc = 3, ncol=1 ,fontsize=20)

    plt.tick_params( axis='both', labelsize=28)
    plt.tick_params( which='major', direction = 'in', length=18, width=1.2, colors='k', left=1, right=1, top =1)
    ax1.tick_params( which='minor', direction = 'in', length=10, width=1.2, colors='k',  left=1, labelbottom=False)

    plt.tight_layout()
    plt.savefig(imname)
    plt.close(fig)
    print("Plot saved in: ", imname)


#label = "AGA"
label = "DE"
outFile = "MergedOuts/mpi_"+label+"_convsChiS_ITS.npy"
outMat = np.load(outFile)
it = np.arange(outMat.shape[0])
best_run = np.argmin(outMat[-1,:])

dir_save = "plots"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
dir_save += "/"
imname = dir_save + label + "Chi2.png"

PlotChi2()                       

