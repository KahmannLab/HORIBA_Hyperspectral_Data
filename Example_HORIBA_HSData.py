from KahmannLab.HORIBA_Hyperspectral_Data import HORIBA_HyperspectralData as HSD
from KahmannLab.HORIBA_Hyperspectral_Data import PCA
from KahmannLab.HORIBA_Hyperspectral_Data import Image_registration as IR
from KahmannLab.HORIBA_Hyperspectral_Data import H5_Editor as H5E
import numpy as np
from importlib import reload
#%% Just run this cell if changes are made in the module HORIBA_HyperspectralData
reload(HSD)
#%% load the hyperspectral data in xml format
folder = r'Y:\d. Optical lab - Microscope user data\Mengru\MS035_PEA2MAPb2I7_old\MS035_PEA2MAPb2I7_old_confocal'
paths = HSD.xml_paths(folder,endswith='.xml',IDs=['ROI-x1'])
#%% save path
save_path = r'Y:\a. Personal folders\Mengru\Data\MS035_PEA2MAPb2I7_old\MS035_confocal_analysis'
#%% load the hyperspectral data with spike removal based on built-in function of HyperSpy
hsdata_PL_1 = HSD.load_xml(paths[0],remove_spikes=True)
hsdata_PL_2 = HSD.load_xml(paths[1],remove_spikes=True)
hsdata_Raman_1 = HSD.load_xml(paths[2],remove_spikes=True)
hsdata_Raman_2 = HSD.load_xml(paths[3],remove_spikes=True)
#%% Extract data arrays, axes, and pixel sizes
data_PL_1 = hsdata_PL_1.data
wl_PL_1 = hsdata_PL_1.axes_manager[2].axis
px_size_PL_1 = hsdata_PL_1.axes_manager[0].scale
data_Raman_1 = hsdata_Raman_1.data
wl_Raman_1 = hsdata_Raman_1.axes_manager[2].axis
px_size_Raman_1 = hsdata_Raman_1.axes_manager[0].scale
#%% load the hyperspectral data in h5 format
h5_folder = r'Y:\d. Optical lab - Microscope user data\Mengru\MS032_POPULAR_correlation\C_printed-S1-1_confocal\H5_files'
paths = H5E.h5_paths(h5_folder, endswith='.h5', keywords=['ROI-1A'])
#%% Extract data arrays, wavelength, and pixel sizes
data, wl, px_size = H5E.data_extract(paths[0], data_loc='/Datas/Data1')
# remove spikes from the data
data_despike = HSD.spike_removal_3d(data,width_threshold=20,prominence_threshold=300)
#%% Visualize the data
HSD.visual_data(data_Raman_1, wl_Raman_1) # plot all spectra in one figure
maps_PL_1 = HSD.plot_maps(data_PL_1, wl_PL_1, px_size_PL_1, data_type='PL') # plot maps to check the features
maps_Raman_1 = HSD.plot_maps(data_Raman_1, wl_Raman_1, px_size_Raman_1, data_type='Raman')
#%% create mask to highlight the features
mask_1 = IR.create_mask(maps_PL_1[0],low_thresh=0,high_thresh=0.3,cleanup=0,plot=True)
mask_2 = IR.create_mask(maps_Raman_1[0],low_thresh=0,high_thresh=0.5,cleanup=0,plot=True)
#%% register the PL and Raman maps before illumination
reg = IR.img_registration(maps_PL_1[0],maps_Raman_1[1],
                          #fixed_mask=mask_1,moving_mask=mask_2,
                          type_of_transform='antsRegistrationSyN[t]',
                          outprefix=save_path+'registration_Raman_1_to_PL_1_',
                          random_seed=24, # for reproducibility
                          return_overlap_flag=True, plot_overlap=True,
                          )
# %% Apply the transformation to the original dataset
cropped_warped_data_Raman_1 = IR.transform2dataset(data_PL_1, data_Raman_1, reg[0],interpolator='nearestNeighbor',
                                           crop_to_overlap_flag=True, bbox=[1,51,0,50])
cropped_data_PL_1 = data_PL_1[1:51,0:50,:]
#%% Noise reduction via PCA
pca_Raman_1, PC_spc_Raman_1 = PCA.sklearn_PCA(cropped_warped_data_Raman_1, ScreePlot=True, n_PCs=20, svd_solver='full')
#%% Plot the spectrum of the PCs
PCA.plot_PCs(PC_spc_Raman_1, wl_Raman_1, component_idx=6)
#%% Plot score map
PCA.score_map(cropped_warped_data_Raman_1, pca_Raman_1, component_index=150, px_size=px_size_Raman_1)
#%% Reconstruct the PL and Raman data
rec_cropped_Raman_1 = PCA.reconstruct_data(cropped_warped_data_Raman_1, pca_Raman_1, component_list=[1,2,9])
# plot reconstructed Raman data
HSD.plot_spectra([cropped_warped_data_Raman_1[30,10,:],rec_cropped_Raman_1[30,10,:]],
                 wl_list=[wl_Raman_1,wl_Raman_1], data_type='Raman',major_locator=100, n_minor_locator=2)
#%% Plot averaged spectrum
avg_PL_1 = HSD.avg_spectrum(cropped_data_PL_1, wl_PL_1, 'PL',jacobian=True,xlabel_PL='Energy (eV)',
                            major_locator=0.2,n_minor_locator=2)
avg_Raman_1 = HSD.avg_spectrum(data_Raman_1, wl_Raman_1, 'Raman',major_locator=200,n_minor_locator=2)
#%% convert wavelength to energy for PL data 1
cropped_jacobian_PL_1, energy_PL_1 = HSD.jacobian_conversion_hsdata(cropped_data_PL_1, wl_PL_1)
#%% Fit the PL spectrum of a single pixel with 3 Gaussian peaks
#for PL data 1
initial_guesses_1 = [1500,1.9 , 0.2, # 1st peak: amplitude, center, width, n=2
                         3000, 2.0,0.05, # 2nd peak: amplitude, center, width, n=3
                         3500, 2.17,0.05]  # 3rd peak: amplitude, center, width
bounds_1 = [[0, 1.8, 0, 0, 1.95, 0, 0, 2.1, 0],
            [np.inf, 1.9, 0.6, np.inf, 2.1, 0.5, np.inf, 2.2, 0.5]]

fit_params_PL1, fit_errors_PL1 = HSD.gaussian_fit(cropped_data_PL_1,xaxis=wl_PL_1,jacobian=True,
                                                  func_name='triple_gaussian', initial_guesses=initial_guesses_1,
                                                  bounds=bounds_1)
HSD.plot_gaussian_fit(cropped_data_PL_1, wl_PL_1, fit_params_PL1,
                      jacobian=True,
                      func_name='triple_gaussian', px_YX=(33,16))
#%%
PL1_intint_gaus1= HSD.intint_gaussian(cropped_data_PL_1, wl_PL_1, params=fit_params_PL1[:,:,0:3],
                                      jacobian=True,
                                      plot=True, px_size=px_size_PL_1,
                                      save_path=None)
#%%
PL1_intint_gaus2= HSD.intint_gaussian(cropped_jacobian_PL_1, energy_PL_1, params=fit_params_PL1[:,:,3:6],
                                      jacobian=False,
                                      plot=True, px_size=px_size_PL_1,
                                      save_path=None)
#%%
PL1_intint_gaus3= HSD.intint_gaussian(cropped_jacobian_PL_1, energy_PL_1, params=fit_params_PL1[:,:,6:9],
                                      jacobian=False,
                                      plot=True, px_size=px_size_PL_1,
                                      save_path=None)
#%% Plot the PL COM map in ROI
com_map_PL_1 = HSD.plot_com_map(cropped_jacobian_PL_1, energy_PL_1,
                                px_size=px_size_PL_1,
                                data_type='PL',
                                int_unit='eV')
#%% Extract the max, min, and mean values of the PL integrated intensity map in ROI [5:55, 5:55]
XY_PL_1 = HSD.select_point(PL1_intint_gaus1)
#%% Mark the points of interest in the PL integrated intensity map for PL data 1
#%% mark the points of interest on the map
HSD.point_marker(com_map_PL_1,px_size_PL_1, XY_PL_1,
                 colorseq='Dark2',cbarlabel='PL center of mass / eV',frac_scalebar=0.23,
                 fontsize=15,markeredgewidth=5,marksize=20,
                 #save_path=save_path+'MS035_03_confocal_PLMap_ROI_x1-cropped2_COM_map-before_marked.png',
                 cbar_sci_notation=False)
#%% Plot the corresponding PL spectra at the points of interest
HSD.Spectrum_extracted(cropped_jacobian_PL_1, energy_PL_1,  XY_PL_1, data_type='PL',
                       major_locator=50, n_minor_locator=2,
                       fontsize=20, labelpad=10, labelsize=15,
                       sci_notation_y=True,linewidth=5)
#%% Plot Raman integrated intensity map in ROI using the registered reconstructed Raman data 1
rec_Raman_intint = HSD.intint_map(rec_cropped_Raman_1, wl_Raman_1, data_type='Raman',
                                  px_size=px_size_Raman_1)