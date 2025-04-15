from KahmannLab.HORIBA_Hyperspectral_Data import HORIBA_HyperspectralData as HSD
from KahmannLab.HORIBA_Hyperspectral_Data import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from importlib import reload
#%% Just run this cell if changes are made in the module HORIBA_HyperspectralData
reload(HSD)
#%% Just run this cell if changes are made in the module PCA
reload(PCA)
#%% load the hyperspectral data in xml format
# here we load the PL and Raman data before and after illumination, named as 1 and 2, respectively
path_PL_1 = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/FormatCov/PLMap_n2_2_0_1 s_532 nm_600 gr_mm_x100_100 µm_300 µm_0_025 %_new_marker.xml'
path_PL_2 = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/FormatCov/PLMap_n2_2_0_1 s_532 nm_600 gr_mm_x100_100 µm_300 µm_0_025 %_new_marker_after_illum.xml'
path_Raman_1 = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/FormatCov/RamanMap_n2_2_0_1 s_785 nm_600 gr_mm_x100_100 µm_300 µm_100 %_new_marker.xml'
path_Raman_2 = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/FormatCov/RamanMap_n2_2_0_1 s_785 nm_600 gr_mm_x100_100 µm_300 µm_100 %_new_marker_after_illum.xml'

#%% save path
save_path = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/Analysis/new marker/'
save_path_pcs = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/Analysis/new marker/PCA/'

#%% load the hyperspectral data
data_PL_1 = HSD.load_xml(path_PL_1)
data_PL_2 = HSD.load_xml(path_PL_2)
data_Raman_1 = HSD.load_xml(path_Raman_1)
data_Raman_2 = HSD.load_xml(path_Raman_2)

#%% remove the data below 0 if needed
data_PL_1 = HSD.remove_negative(data_PL_1)
data_PL_2 = HSD.remove_negative(data_PL_2)
data_Raman_1 = HSD.remove_negative(data_Raman_1)
data_Raman_2 = HSD.remove_negative(data_Raman_2)
#%% Plot integrated intensity maps
# PL integrated intensity map before illumination
intint_PL_1 = HSD.intint_map(data_PL_1,'PL', (595, 635))
#%% Raman integrated intensity map before illumination
intint_Raman_1 = HSD.intint_map(data_Raman_1,'Raman',(40, 100))
#%% PL integrated intensity map after illumination
intint_PL_2 = HSD.intint_map(data_PL_2, 'PL',(500,900))
#%% Raman integrated intensity map after illumination
intint_Raman_2 = HSD.intint_map(data_Raman_2,'Raman',(40, 70))
#%% Create mask to highlight the area of fiducial markers
mask_PL_1 = HSD.create_mask(intint_PL_1, 80, 100,100)
mask_PL_2 = HSD.create_mask(intint_PL_2, 50, 100,100)
mask_Raman_1 = HSD.create_mask(intint_Raman_1, 170, 100,100)
mask_Raman_2 = HSD.create_mask(intint_Raman_2, 175, 100,100)
# check the mask
plt.imshow(mask_PL_1)
plt.colorbar()
plt.show()

#%% Image registration based on the masked PL and Raman maps
outprefix = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/Analysis/new marker/'
# i gives the random seed for the transformation
for i in range(26,27,1):
    transf_1, warped_mov_1 = HSD.ants_registration(mask_PL_1, mask_Raman_1,'Translation',random_seed=i,
                                                   outprefix=outprefix+'Raman1_')
    #plt.savefig(save_path+'Masked_PL_Raman_maps_registration_before_rs{}.png'.format(i),transparent=True)
    plt.show()
    plt.close()

#%% Image registration based on masked PL maps before and after illumination
# PL map based on the PL data before illumination is the fixed image for all registrations
for i in range(45,46,1):
    transf_2, warped_mov_2 = HSD.ants_registration(mask_PL_1, mask_PL_2,'Translation',random_seed=i,
                                                   outprefix=outprefix+'PL2_') # register PL map before illumination to that after illumination
    #plt.savefig(save_path+'Masked_maps_registration_before_after_rs{}.png'.format(i),transparent=True)
    plt.show()
    plt.close()

#%% Image registration based on the PL before illumination and Raman maps after illumination
for i in range(38,39,1):
    transf_3, warped_mov_3 = HSD.ants_registration(mask_PL_1, mask_Raman_2,'Translation',random_seed=i,
                                                   outprefix=outprefix+'Raman2_') # register PL map before illumination to that after illumination
    #plt.savefig(save_path+'Masked_PL_Raman_maps_registration_before_after_rs{}.png'.format(i),transparent=True)
    plt.show()

# %% Apply the transformation to the original dataset
warped_data_Raman_1 = HSD.transform2hsi(data_PL_1, data_Raman_1, transf_1)
# Apply the transformation to the original PL2 dataset (registered PL after illumination)
warped_data_PL_2 = HSD.transform2hsi(data_PL_1, data_PL_2, transf_2)
# Apply the transformation to the original Raman2 dataset (registered Raman after illumination)
warped_data_Raman_2 = HSD.transform2hsi(data_PL_1, data_Raman_2, transf_3)

# %% Noise reduction via PCA
# default algorithm is SVD (singular value decomposition)
data_PL_1.decomposition(algorithm='SVD', center='signal')
data_Raman_1.decomposition(algorithm='SVD', center='signal')
data_PL_2.decomposition(algorithm='SVD', center='signal')
data_Raman_2.decomposition(algorithm='SVD', center='signal')

#%% Plot the explained variance ratio (EVR) for the Raman data
PCA.plot_evr(data_Raman_1, 50)
#%% Plot the spectrum of the PCs
for i in range(10):
    PCA.PC_spc(data_PL_1, i, 'PL',
               save_path=None)
               #save_path=save_path_pcs+'before_PL_PC{}_spectrum.png'.format(i+1))
#%% Reconstruct the PL and Raman data
rec_data_PL_1 = PCA.rec_data(data_PL_1, 20) # using the first 20 PCs
rec_data_Raman_1 = PCA.rec_data(data_Raman_1, 20) # using the first 20 PCs
rec_data_PL_2 = PCA.rec_data(data_PL_2, 35) #using the first 35 PCs
rec_data_Raman_2 = PCA.rec_data(data_Raman_2, 20) # using the first 20 PCs
#%% warp reconstructed Raman data 1
warped_rec_data_Raman_1 = HSD.transform2hsi(data_PL_1, rec_data_Raman_1, transf_1, interpolator='nearestNeighbor')
# warp reconstructed PL data
warped_rec_data_PL_2 = HSD.transform2hsi(data_PL_1, rec_data_PL_2, transf_2, interpolator='nearestNeighbor')
# warp reconstructed Raman data 2
warped_rec_data_Raman_2 = HSD.transform2hsi(data_PL_1, rec_data_Raman_2, transf_3, interpolator='nearestNeighbor')

#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks
#for PL data 1
initial_guesses_1 = [15, 575, 10, # 1st peak: amplitude, center, width, n=2
                         50, 620, 10, # 2nd peak: amplitude, center, width, n=3
                         35, 650, 10]  # 3rd peak: amplitude, center, width

PL1_gaus_params,_ = HSD.gaussian_fit(data_PL_1, func_name='triple_gaussian', initial_guesses=initial_guesses_1)
#%%
PL1_intint_gaus1= HSD.plot_intint_gaussian(data_PL_1, params=PL1_gaus_params[:,:,0:3],
                                           save_path=None)
                                           #save_path=save_path+'PL_Integrated_intensity_map_Gaus1.png')
#%%
PL1_intint_gaus2= HSD.plot_intint_gaussian(data_PL_1, params=PL1_gaus_params[:,:,3:6],
                                           save_path=None)
#%%
PL1_intint_gaus3= HSD.plot_intint_gaussian(data_PL_1, params=PL1_gaus_params[:,:,6:9],
                                           save_path=None)
#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks (for reconstructed PL data 1)
rec_PL1_intint_gaus1 = HSD.plot_intint_gaussian(rec_data_PL_1, params=PL1_gaus_params[:,:,0:3],sum_threshold=4000)
rec_PL1_intint_gaus2 = HSD.plot_intint_gaussian(rec_data_PL_1, params=PL1_gaus_params[:,:,3:6],sum_threshold=4000)
rec_PL1_intint_gaus3 = HSD.plot_intint_gaussian(rec_data_PL_1, params=PL1_gaus_params[:,:,6:9],sum_threshold=4000)
#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks
#for PL data 2
initial_guesses_2 = [15, 575, 10, # 1st peak: amplitude, center, width, n=2
                            35, 620, 10, # 2nd peak: amplitude, center, width, n=3
                            20, 650, 10] # 3rd peak: amplitude, center, width, n=4
PL2_gaus_params,_ = HSD.gaussian_fit(data_PL_2, func_name='triple_gaussian', initial_guesses=initial_guesses_2)
#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks (for reconstructed PL data 2)
rec_PL2_intint_gaus1 = HSD.plot_intint_gaussian(rec_data_PL_2, params=PL2_gaus_params[:,:,0:3],sum_threshold=4000)
rec_PL2_intint_gaus2 = HSD.plot_intint_gaussian(rec_data_PL_2, params=PL2_gaus_params[:,:,3:6],sum_threshold=4000)
rec_PL2_intint_gaus3 = HSD.plot_intint_gaussian(rec_data_PL_2, params=PL2_gaus_params[:,:,6:9],sum_threshold=4000)
#%% Register the reconstructed integrated intensity map of the individual Gaussian peak for PL data 2 to that for PL data 1
warped_rec_PL2_intint_gaus1 = HSD.transform2map(rec_PL1_intint_gaus1, rec_PL2_intint_gaus1, transf_2)
warped_rec_PL2_intint_gaus2 = HSD.transform2map(rec_PL1_intint_gaus2, rec_PL2_intint_gaus2, transf_2)
warped_rec_PL2_intint_gaus3 = HSD.transform2map(rec_PL1_intint_gaus3, rec_PL2_intint_gaus3, transf_2)

#%% Visulize a ROI of the 1st integrated intensity map over the entire spectral range using reconstructed PL data 1
HSD.plot_intint_gaussian(rec_data_PL_1, PL1_gaus_params[:,:,0:3], ROI=(5,5,50,50))
#%% Plot the PL COM map in ROI
com_map_PL_1 = HSD.plot_com_map(rec_data_PL_1, data_type='PL', params_ROI=(5,5,50,50))
#%%
com_map_PL_2 = HSD.plot_com_map(rec_data_PL_2, processed_data=warped_rec_data_PL_2, data_type='PL', params_ROI=(5,5,50,50))
#%% Extract the max, min, and mean values of the PL integrated intensity map in ROI [5:55, 5:55]
YX_PL1_guas1 = [HSD.coord_extract(rec_PL1_intint_gaus1[5:55,5:55], 'max'),
                HSD.coord_extract(rec_PL1_intint_gaus1[5:55,5:55], 'min'),
                HSD.coord_extract(rec_PL1_intint_gaus1[5:55,5:55], 'median')]
#%% Mark the points of interest in the PL integrated intensity map for PL data 1
HSD.point_marker(rec_PL1_intint_gaus1[5:55,5:55], rec_data_PL_1, YX_PL1_guas1, 'PL integrated intensity (a.u.)')
#%% Plot the corresponding PL spectra at the points of interest
HSD.Spectrum_extracted(rec_data_PL_1, YX_PL1_guas1, data_type='PL',spc_labels=['max','min','median'],
                       processed_data=rec_data_PL_1.data[5:55,5:55,:])
#%% Extract the corresponding Raman spectra at the points of interest extracted from the PL integrated intensity map for PL data 1
HSD.Spectrum_extracted(rec_data_Raman_1, YX_PL1_guas1, data_type='Raman',spc_labels=['max','min','median'],
                       major_locator=500, n_minor_locator=5,
                       processed_data=warped_rec_data_Raman_1[5:55,5:55,:])
#%% Plot Raman integrated intensity map in ROI using the registered reconstructed Raman data 1
warped_rec_Raman_intint_1 = HSD.intint_map(rec_data_Raman_1, 'Raman',(70, 100),warped_rec_data_Raman_1[5:55,5:55,:])
#%% Plot Raman (relative) intensity map at 100 cm^-1 in ROI using the registered reconstructed Raman data 1
warped_rec_Raman_1_int_1 = HSD.int_map(rec_data_Raman_1, 1378, 'Raman',processed_data=warped_rec_data_Raman_1[5:55,5:55,:])
warped_rec_Raman_1_int_2 = HSD.int_map(rec_data_Raman_1, 90, 'Raman',processed_data=warped_rec_data_Raman_1[5:55,5:55,:])
rel_int_Raman_1 = warped_rec_Raman_1_int_2/warped_rec_Raman_1_int_1
#%% correlation map between the PL integrated intensity over gaussian peak 1 and relative Raman intensity at 90 cm^-1
HSD.plot_2d_hist([rec_PL1_intint_gaus3[5:55,5:55], rel_int_Raman_1],
                 labels=['PL integrated intensity (a.u.)','Relative Raman intensity (a.u.)'])
#%% Plot PL average spectrum
avg_PL_1 = HSD.avg_spectrum(data_PL_1, 'PL')
#%% Plot PL average spectrum PL data 2
avg_PL_2 = HSD.avg_spectrum(data_PL_2, 'PL')
#%% Plot the average PL spectrum before & after on the same figure
spcs = [avg_PL_1, avg_PL_2]
axes = [data_PL_1.axes_manager[2].axis, data_PL_2.axes_manager[2].axis]
#%%
HSD.plot_spectra(spcs, axes, 'PL',
                 label_list=['PL before illumination','PL after illumination'])
#%% Plot averaged Raman spectrum
avg_Raman_1 = HSD.avg_spectrum(data_Raman_1, 'Raman',major_locator=500, n_minor_locator=5)
avg_Raman_2 = HSD.avg_spectrum(data_Raman_2, 'Raman',major_locator=500, n_minor_locator=5)
#%%
spcs_Raman = [avg_Raman_1, avg_Raman_2]
axes_Raman = [data_Raman_1.axes_manager[2].axis, data_Raman_2.axes_manager[2].axis]
HSD.plot_spectra(spcs_Raman, axes_Raman, 'Raman',major_locator=500, n_minor_locator=5,
                 label_list=['Raman before illumination','Raman after illumination'])
#%% Find the maxima in the average PL spectrum
HSD.find_maxima(data_Raman_1, avg_Raman_1, 'Raman', 0.3,25)
#%% Plot a histogram of the PL integrated intensity map
HSD.plot_hist(rec_PL1_intint_gaus1[5:55,5:55], 'PL integrated intensity (a.u.)', bins=500,
              bins_range=(1,1000), spread=True)
#%% Compare the histograms of the PL integrated intensity maps before and after illumination
hist_Gaus1_PL1 = [rec_PL1_intint_gaus1[5:55,5:55], warped_rec_PL2_intint_gaus1[5:55,5:55]]
hist_Gaus2_PL1 = [rec_PL1_intint_gaus2[5:55,5:55], warped_rec_PL2_intint_gaus2[5:55,5:55]]
hist_Gaus3_PL1 = [rec_PL1_intint_gaus3[5:55,5:55], warped_rec_PL2_intint_gaus3[5:55,5:55]]
HSD.plot_hist(hist_Gaus1_PL1, labels=['PL before illumination','PL after illumination'],
              x_label='PL integrated intensity (a.u.)', bins=500,
              bins_range=(1,1000), spread=True,
              save_path=None)
              #save_path=save_path+'Gaus1_PL_histogram.png')
HSD.plot_hist(hist_Gaus2_PL1, labels=['PL before illumination','PL after illumination'],
               x_label='PL integrated intensity (a.u.)', bins=500,
                bins_range=(1,3000), spread=True,
                save_path=None)
                #save_path=save_path+'Gaus2_PL_histogram.png')
HSD.plot_hist(hist_Gaus3_PL1, labels=['PL before illumination','PL after illumination'],
               x_label='PL integrated intensity (a.u.)', bins=500,
                bins_range=(1,5000), spread=True,
                save_path=None)
                #save_path=save_path+'Gaus3_PL_histogram.png')

#%% Gaussian fit for PL spectrum of a single pixel, e.g. maximum in the PL integrated intensity map 1
# for PL data 1
HSD.plot_gaussian_fit(rec_data_PL_1, PL1_gaus_params, func_name='triple_gaussian', px_YX=YX_PL1_guas1[0])
#%% Plot loading maps to identify the spatial distribution of the individual Gaussian peaks
# loading of the 2nd PC: which represents a distribution of the emission at 650 nm
PCA.loading_map(rec_data_PL_1,2)

#%% animation to show the reconstructed PL data 1
PCA.rec_spc_animation(data_PL_1, px=(15,25), start_nPCs=200, end_nPCs=20, data_type='PL',
                      save_path=save_path+'PL1_reconstructed_animation.gif')
#%% animation to show the reconstructed Raman data 1
PCA.rec_spc_animation(data_Raman_1, px=(15,25), start_nPCs=200, end_nPCs=20, data_type='Raman',
                      major_locator=500, n_minor_locator=5,
                      save_path=save_path+'Raman1_reconstructed_animation.gif')
#%% Plot the white light reflection image
import pandas as pd
img_Raman_1_path = 'F:/TUC_Data/Data_Microscope/MS013_PEAMAPbI-n2-fre/MS013_241104/FormatCov/RamanMap_n2_2_0_1 s_785 nm_600 gr_mm_x100_100 µm_300 µm_100 %_new_marker.txt'
img_Raman_1 = pd.read_csv(img_Raman_1_path, sep='\t', header=0, index_col=0)
#%%
HSD.plot_img(img_Raman_1)