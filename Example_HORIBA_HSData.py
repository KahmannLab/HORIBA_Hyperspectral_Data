import HORIBA_HyperspectralData as hsda
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

#%% Just run this cell if changes are made in the module HORIBA_HyperspectralData
from importlib import reload
reload(hsda)
import HORIBA_HyperspectralData as hsda
#%% load the hyperspectral data in xml format
# here we load the PL and Raman data before and after illumination, named as 1 and 2, respectively
path_PL_1 = 'C:/Data_Microscope/241104_MS_PMPI2/FormatCov/PLMap_n2_2_0_1 s_532 nm_600 gr_mm_x100_100 µm_300 µm_0_025 %_new_marker.xml'
path_PL_2 = 'C:/Data_Microscope/241104_MS_PMPI2/FormatCov/PLMap_n2_2_0_1 s_532 nm_600 gr_mm_x100_100 µm_300 µm_0_025 %_new_marker_after_illum.xml'
path_Raman_1 = 'C:/Data_Microscope/241104_MS_PMPI2/FormatCov/RamanMap_n2_2_0_1 s_785 nm_600 gr_mm_x100_100 µm_300 µm_100 %_new_marker.xml'
path_Raman_2 = 'C:/Data_Microscope/241104_MS_PMPI2/FormatCov/RamanMap_n2_2_0_1 s_785 nm_600 gr_mm_x100_100 µm_300 µm_100 %_new_marker_after_illum.xml'

#%% save path
save_path = 'C:/Data_Microscope/241104 PEA2MAPb2I7_degradation/Analysis/new marker/'
save_path_pcs = 'C:/Data_Microscope/241104 PEA2MAPb2I7_degradation/Analysis/new marker/PCA/'

#%% load the hyperspectral data
data_PL_1 = hsda.load_xml(path_PL_1)
data_PL_2 = hsda.load_xml(path_PL_2)
data_Raman_1 = hsda.load_xml(path_Raman_1)
data_Raman_2 = hsda.load_xml(path_Raman_2)

#%% remove the data below 0 if needed
data_PL_1 = hsda.remove_negative(data_PL_1)
data_PL_2 = hsda.remove_negative(data_PL_2)
data_Raman_1 = hsda.remove_negative(data_Raman_1)
data_Raman_2 = hsda.remove_negative(data_Raman_2)

#%% Plot integrated intensity maps
# PL integrated intensity map before illumination
intint_PL_1 = hsda.intint_map(data_PL_1,'PL', (595, 635), None)
#plt.savefig(save_path+'PL_Integrated_intensity_map_595_635_before.png',transparent=True,dpi=300)
plt.show()
#%% Raman integrated intensity map before illumination
intint_Raman_1 = hsda.intint_map(data_Raman_1,'Raman',(40, 100),None)
#plt.savefig(save_path+'Raman_Integrated_intensity_map_40_100_before.png',transparent=True,dpi=300)
plt.show()
#%% PL integrated intensity map after illumination
intint_PL_2 = hsda.intint_map(data_PL_2, 'PL',(500,900), warped_data=None)
#plt.savefig(save_path+'PL_Integrated_intensity_map_595_635_after.png',transparent=True,dpi=300)
plt.show()
#%% Raman integrated intensity map after illumination
intint_Raman_2 = hsda.intint_map(data_Raman_2,'Raman',(40, 70),None)
#plt.savefig(save_path+'Raman_Integrated_intensity_map_40_100_after.png',transparent=True,dpi=300)
plt.show()

#%% Create mask to highlight the area of fiducial markers
mask_PL_1 = hsda.create_mask(intint_PL_1[0], 80, 100,100)
mask_PL_2 = hsda.create_mask(intint_PL_2[0], 50, 100,100)
mask_Raman_1 = hsda.create_mask(intint_Raman_1[0], 170, 100,100)
mask_Raman_2 = hsda.create_mask(intint_Raman_2[0], 175, 100,100)
# check the mask
plt.imshow(mask_PL_1)
plt.colorbar()
plt.show()

#%% Image registration based on the masked PL and Raman maps
# i gives the random seed for the transformation
for i in range(26,27,1):
    transf_1, warped_mov_1 = hsda.ants_registration(mask_PL_1, mask_Raman_1,'Translation',i)
    #plt.savefig(save_path+'Masked_PL_Raman_maps_registration_before_rs{}.png'.format(i),transparent=True)
    plt.show()
    plt.close()

#%% Image registration based on masked PL maps before and after illumination
# PL map based on the PL data before illumination is the fixed image for all registrations
for i in range(45,46,1):
    transf_2, warped_mov_2 = hsda.ants_registration(mask_PL_1, mask_PL_2,'Translation',i) # register PL map before illumination to that after illumination
    #plt.savefig(save_path+'Masked_maps_registration_before_after_rs{}.png'.format(i),transparent=True)
    plt.show()
    plt.close()

#%% Image registration based on the PL before illumination and Raman maps after illumination
for i in range(38,39,1):
    transf_3, warped_mov_3 = hsda.ants_registration(mask_PL_1, mask_Raman_2,'Translation',i)
    #plt.savefig(save_path+'Masked_PL_Raman_maps_registration_before_after_rs{}.png'.format(i),transparent=True)
    plt.show()

# %% Apply the transformation to the original dataset
warped_data_Raman_1 = hsda.transform2hsi(data_PL_1, data_Raman_1, transf_1)
# Apply the transformation to the original PL2 dataset (registered PL after illumination)
warped_data_PL_2 = hsda.transform2hsi(data_PL_1, data_PL_2, transf_2)
# Apply the transformation to the original Raman2 dataset (registered Raman after illumination)
warped_data_Raman_2 = hsda.transform2hsi(data_PL_1, data_Raman_2, transf_3)

# %% Noise reduction via PCA
# default algorithm is SVD (singular value decomposition)
data_PL_1.decomposition(True)
data_Raman_1.decomposition(True)
data_PL_2.decomposition(True)
data_Raman_2.decomposition(True)

#%% Plot the explained variance ratio (EVR) for the Raman data
hsda.plot_evr(data_Raman_1, 50)
#plt.savefig(save_path_pcs+'Raman_EVR_before.png',transparent=True,dpi=300)
plt.show()

#%% Plot the spectrum of the PCs
for i in range(10):
    hsda.PC_spc(data_PL_1, i, 'PL')
    #plt.savefig(save_path_pcs+'after_PL_PC{}_spectrum.png'.format(i+1),transparent=True,dpi=300)
    plt.show()

#%% Reconstruct the PL and Raman data
rec_data_PL_1 = hsda.rec_data(data_PL_1, 20) # using the first 20 PCs
rec_data_Raman_1 = hsda.rec_data(data_Raman_1, 20) # using the first 20 PCs
rec_data_PL_2 = hsda.rec_data(data_PL_2, 35) #using the first 35 PCs
rec_data_Raman_2 = hsda.rec_data(data_Raman_2, 20) # using the first 20 PCs
#%% warp reconstructed Raman data 1
warped_rec_data_Raman_1 = hsda.transform2hsi(data_PL_1, rec_data_Raman_1, transf_1, interpolator='nearestNeighbor')
# warp reconstructed PL data
warped_rec_data_PL_2 = hsda.transform2hsi(data_PL_1, rec_data_PL_2, transf_2, interpolator='nearestNeighbor')
# warp reconstructed Raman data 2
warped_rec_data_Raman_2 = hsda.transform2hsi(data_PL_1, rec_data_Raman_2, transf_3, interpolator='nearestNeighbor')

#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks
#for PL data 1
initial_guesses_1 = [15, 575, 10, # 1st peak: amplitude, center, width, n=2
                         50, 620, 10, # 2nd peak: amplitude, center, width, n=3
                         35, 650, 10] # 1st peak: amplitude, center, width

PL1_intint_gaus1, PL1_intint_gaus2, PL1_intint_gaus3 = hsda.intint_gaussian(data_PL_1, initial_guesses=initial_guesses_1, sum_int_threshold=4000)

#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks (for reconstructed PL data 1)
rec_PL1_intint_gaus1, rec_PL1_intint_gaus2, rec_PL1_intint_gaus3 = hsda.intint_gaussian(data_PL_1, processed_data=rec_data_PL_1, initial_guesses=initial_guesses_1, sum_int_threshold=4000)
#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks
#for PL data 2
initial_guesses_2 = [15, 575, 10, # 1st peak: amplitude, center, width, n=2
                            35, 620, 10, # 2nd peak: amplitude, center, width, n=3
                            20, 650, 10] # 3rd peak: amplitude, center, width, n=4

PL2_intint_gaus1, PL2_intint_gaus2, PL2_intint_gaus3 = hsda.intint_gaussian(data_PL_2, initial_guesses=initial_guesses_2, sum_int_threshold=4000)
#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks (for reconstructed PL data 2)
rec_PL2_intint_gaus1, rec_PL2_intint_gaus2, rec_PL2_intint_gaus3 = hsda.intint_gaussian(data_PL_2, processed_data=rec_data_PL_2, initial_guesses=initial_guesses_2, sum_int_threshold=4000)
#%% Register the reconstructed integrated intensity map of the individual Gaussian peak for PL data 2 to that for PL data 1
warped_rec_PL2_intint_gaus1 = hsda.transform2map(rec_PL1_intint_gaus1, rec_PL2_intint_gaus1, transf_2)
warped_rec_PL2_intint_gaus2 = hsda.transform2map(rec_PL1_intint_gaus2, rec_PL2_intint_gaus2, transf_2)
warped_rec_PL2_intint_gaus3 = hsda.transform2map(rec_PL1_intint_gaus3, rec_PL2_intint_gaus3, transf_2)

#%% Check the registration
hsda.check_registration(rec_PL1_intint_gaus1, warped_rec_PL2_intint_gaus1)

#%% plot the integrated intensity map of the 1st Gaussian peak over the entire spectral range for PL data 1
hsda.plot_intint_gaussian(data_PL_1, PL1_intint_gaus1)
#plt.savefig(save_path+'before_PL_Integrated_intensity_map_Gaus1.png',transparent=True,dpi=300)
plt.show()
#%% plot the ROI extracted from the integrated intensity map of the 1st Gaussian peak over the entire spectral range using reconstructed PL data 1
hsda.plot_intint_gaussian(data_PL_1, rec_PL1_intint_gaus1, params_ROI=(5,5,50,50))
#plt.savefig(save_path+'before_Rec_PL_Integrated_intensity_map_n=2_Gaus_ROI5_5_50.png',transparent=True,dpi=300)
plt.show()

#%% Plot the PL COM map in ROI
com_map_PL_1 = hsda.plot_com_map(rec_data_PL_1, data_type='PL',params_ROI=(5,5,50,50))
#plt.savefig(save_path+'before_rec_PL_COM_map_ROI5_5_50.png',transparent=True,dpi=300)
plt.show()

#%%
com_map_PL_2 = hsda.plot_com_map(rec_data_PL_2, processed_data=warped_rec_data_PL_2, data_type='PL',params_ROI=(5,5,50,50))
#plt.savefig(save_path+'after_registered_rec_PL_COM_map_ROI5_5_50.png',transparent=True,dpi=300)
plt.show()

#%% Mark the points of interest in the PL com map 1
# select 3 points
XY = [(18,15),(38,35),(28,49)] # (x,y) for com_map_PL
fig_com,ax_com = hsda.point_marker(com_map_PL_1, rec_data_PL_1, XY, 'PLCOM')
#plt.savefig(save_path+'before_PL_COM_map_3Points.png',transparent=True,dpi=300)
plt.show()
#%% mark the points of interest in the ROI of PL COM map 1
XY_ROI = [(x-5,y-5) for x,y in XY]
fig_com,ax_com = hsda.point_marker(com_map_PL_1[5:55,5:55], rec_data_PL_1, XY_ROI, 'PLCOM')
#plt.savefig(save_path+'before_PL_COM_map_3Points_ROI.png',transparent=True,dpi=300)
plt.show()

#%% Extract the spectrum at the points of interest
hsda.Spectrum_extracted(rec_data_PL_1, XY, data_type='PL')
#plt.savefig(save_path+'before_rec_PL_spectra_3Points.png',transparent=True,dpi=300)
plt.show()

#%% Plot Raman integrated intensity map in ROI using the registered reconstructed Raman data 1
warped_rec_Raman_intint_1 = hsda.intint_map(rec_data_Raman_1, 'Raman',(70, 100),warped_rec_data_Raman_1[5:55,5:55,:])
plt.show()
#%% Plot Raman (relative) intensity map at 100 cm^-1 in ROI using the registered reconstructed Raman data 1
warped_rec_norm_data_Raman_1 = warped_rec_data_Raman_1/np.max(warped_rec_data_Raman_1,axis=2)[:,:,np.newaxis]
warped_rec_Raman_int_1 = hsda.int_map(rec_data_Raman_1, 100, 'RamanRatio',warped_data=warped_rec_norm_data_Raman_1[5:55,5:55,:])
#plt.savefig(save_path+'before_warped_rec_norm_{}_Intensity_map_at_{}_ROI.png'.format('Raman',100),transparent=True,dpi=300)
#plt.savefig(save_path+'before_warped_rec_{}_Intensity_map_at_{}_ROI.png'.format('Raman',100),transparent=True,dpi=300)
plt.show()

#%% Plot PL average spectrum
avg_PL_1 = hsda.avg_spectrum(data_PL_1, 'PL')
#plt.savefig(save_path+'PL_Average_spectrum_before.png',transparent=True,dpi=300)
plt.show()
#%% Plot PL average spectrum PL data 2
avg_PL_2 = hsda.avg_spectrum(data_PL_2, 'PL')
#plt.savefig(save_path+'PL_Average_spectrum_after.png',transparent=True,dpi=300)
plt.show()
#%% Plot the average PL spectrum before & after on the same figure
hsda.spectra_compare(avg_PL_1[0], avg_PL_2[0], 'PL', data_PL_1, data_PL_2)
#plt.savefig(save_path+'PL_AvgSpectrum_before_after.png',transparent=True,dpi=300)
plt.show()

#%% Find the maxima in the average PL spectrum
hsda.find_maxima(data_PL_1, avg_PL_1[0], 'PL', 5,10)
#plt.savefig(save_path+'Raman_AvgSpectrum_peaks_before.png',transparent=True)
plt.show()
