import numpy as np


def concat(label,n_ranks,dir_save, dir_merged):

	ranks = n_ranks
	arr_conv = 0
	arr_sols = 0
	arr_time = 0
	for rank in range(ranks):
		filename_conv   = dir_save + "mpi_" + label + "_convsChiS_"  + str(rank) + "_ITS.npy"    
		filename_sols   = dir_save + "mpi_" + label + "_sols_"       + str(rank) + "_ITS.npy"    
		filename_time   = dir_save + "mpi_" + label + "_time_"       + str(rank) + "_ITS.npy"

		if(rank==0):
	   		arr_conv = np.load(filename_conv)
	   		arr_sols = np.load(filename_sols)
	   		arr_time = np.load(filename_time)
	   	#print("arr shape: ",np.shape(arr))
		else:
	   		newarr_conv = np.load(filename_conv)
	   		newarr_sols = np.load(filename_sols)
	   		newarr_time = np.load(filename_time)
	   		#print("newarr shape:", np.shape(newarr))
	   		arr_conv = np.concatenate((arr_conv,newarr_conv), axis=1)
	   		arr_sols = np.concatenate((arr_sols,newarr_sols), axis=1)
	   		arr_time = np.concatenate((arr_time,newarr_time), axis=0)
		
	filename_conv   = dir_merged + "mpi_" + label + "_convsChiS_ITS.npy"    
	filename_sols   = dir_merged + "mpi_" + label + "_sols_ITS.npy"    
	filename_time   = dir_merged + "mpi_" + label + "_time_ITS.npy"

	print("Shape of "+label+" conv merged array: ", np.shape(arr_conv))
	print("Shape of "+label+" sols merged array: ", np.shape(arr_sols))
	print("Shape of "+label+" time merged array: ", np.shape(arr_time))

	np.save(filename_conv, arr_conv)
	np.save(filename_sols, arr_sols)
	np.save(filename_time, arr_time)

	print(label+" merged conv array saved in: ", filename_conv)
	print(label+" merged sols array saved in: ", filename_sols)
	print(label+" merged time array saved in: ", filename_time)

