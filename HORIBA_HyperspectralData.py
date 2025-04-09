import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import ants
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import os

#%% extract the xml file paths
def xml_paths(folder_path, endswith='.xml'):
    """
    Extract the sif file paths from the folder
    :param folder_path (str): the path to the folder
    :return: sif_paths (list): the list of the sif file paths
    """
    xml_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(endswith):
                xml_paths.append(os.path.join(root, file))
    return xml_paths
#%% Load the hyperspectral data in an .xml file exported(or saved) from LabSpec(HORIBA) software
def load_xml(file_path):
    '''
    Load the hyperspectral data in an .xml file
    :param file_path (str): the path of the hyperspectral data file
    :return:
    '''
    data = hs.load(file_path, reader='JobinYvon')
    return data
#%% Remove the values below 0 from the raw data
def remove_negative(data):
    """
    Modify the raw data by removing the values below 0
    :param  data (LumiSpectrum): the original hyperspectral data
    :return: data (LumiSpectrum): the modified hyperspectral data
    """
    data.data[data.data < 0] = 0
    return data
#%% Plot an integrated intensity map
def intint_map(data, data_type, spectral_range=None, processed_data=None,
               savefig=False, figname=None, savefile=False, filename=None, savepath=None):
    """
    Plot an integrated intensity map over the given wavelength or wavenumber range
    :param data (LumiSpectrum): hyperspectral data
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param warped_data (np.ndarray)(optional): the registered data
    :return: intint (np.ndarray): the integrated intensity map
    """
    Spectr = data.axes_manager[2].axis
    if spectral_range is not None:
        index1 = abs(Spectr - spectral_range[0]).argmin()
        index2 = abs(Spectr - spectral_range[1]).argmin()
    else:
        index1 = 0
        index2 = data.data.shape[2]
    if processed_data is not None:
        intint = processed_data[:,:,index1:index2].sum(axis=2)
    else:
        intint = data.data[:,:,index1:index2].sum(axis=2)

    scalebar = ScaleBar(data.axes_manager[0].scale, "um", length_fraction=0.13, location='lower right',
                        box_color=None, color='white', frameon=False, width_fraction=0.02, font_properties={'size': 12,
                                                                                                            'weight': 'bold',
                                                                                                            'math_fontfamily': 'dejavusans'})
    # Use histogram to get a better contrast
    hist, bins = np.histogram(intint.flatten(), bins=500)

    # find the bin-edges for the range from the maximal counts of 5% to the maximal counts of 95% in the histogram
    cum_counts = np.cumsum(hist)
    tot_counts = np.sum(hist)
    tot_counts_5 = tot_counts * 0.05
    tot_counts_95 = tot_counts * 0.95
    bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]

    # Plot the map
    fig,ax = plt.subplots()
    cmap = ax.imshow(intint, vmin=bin_edges[0],vmax=bin_edges[-1],cmap='viridis') # use the bin-edges as the limits for correcting the color scale
    ax.set_axis_off()
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(cmap, ax=ax, format=fmt)
    cbar.set_label('{} integrated intensity (counts)'.format(data_type))
    ax.add_artist(scalebar)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
    plt.show()

    if savefile:
        if filename is None:
            print('Please provide a file name')
        else:
            np.savetxt(savepath+filename+'.txt', intint)
    return intint

#%%plot an intensity map(relative intensity map) at the given wavelength or wavenumber
def int_map(original_data, wavelength,data_type,processed_data=None,
            savefig=False, figname=None, savefile=False, filename=None, savepath=None):
    """
    Plot an intensity map at the given wavelength or wavenumber
    :param original_data (LumiSpectrum): original/only-reconstructed hyperspectral data,
            if warped_data is given, the original data is just used for providing the spectral axis information
    :param wavelength (float): the wavelength or wavenumber
    :param data_type (str): the type of the data: 'PL', 'Raman', 'PLRatio', 'RamanRatio'
    :param processed_data (np.ndarray)(optional): the registered data or any data different from the original data
    :return: int_map (np.ndarray): the intensity map at the given wavelength or wavenumber,
                fig (matplotlib.figure.Figure): the figure object,
                ax (matplotlib.axes._axes.Axes): the axes object,
    """
    Spectr = original_data.axes_manager[2].axis
    if processed_data is not None:
        int_map = processed_data[:,:,abs(Spectr-wavelength).argmin()]
    else:
        int_map = original_data.data[:,:,abs(Spectr-wavelength).argmin()]
    scalebar = ScaleBar(original_data.axes_manager[0].scale, "um", length_fraction=0.13, location='lower right',
                        box_color=None, color='white', frameon=False, width_fraction=0.02, font_properties={'size': 12,
                                                                                                        'weight': 'bold',
                                                                                                        'math_fontfamily': 'dejavusans'})
    # Use histogram to avoid the bad pixel in order to get a better ratio map
    hist, bins = np.histogram(int_map.flatten(), bins=500)

    # find the bin-edges for the range from the maximal counts of 5% to the maximal counts of 95% in the histogram
    cum_counts = np.cumsum(hist)
    tot_counts = np.sum(hist)
    tot_counts_5 = tot_counts * 0.05
    tot_counts_95 = tot_counts * 0.95
    bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]

    fig, ax = plt.subplots()
    cmap = ax.imshow(int_map,vmin=bin_edges[0], vmax=bin_edges[-1])
    ax.set_axis_off()
    ax.add_artist(scalebar)
    if data_type == 'PL':
        cbarlabel = 'PL intensity (counts)'
    elif data_type == 'Raman':
        cbarlabel = 'Raman intensity (counts)'
    elif data_type == 'PLRatio':
        cbarlabel = 'Normalized PL intensity'
    elif data_type == 'RamanRatio':
        cbarlabel = 'Normalized Raman intensity'
    cbar=fig.colorbar(cmap, ax=ax)
    cbar.set_label(cbarlabel, fontsize=12, labelpad=10)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
    plt.show()

    if savefile:
        if filename is None:
            print('Please provide a file name')
        else:
            np.savetxt(savepath+filename+'.txt', int_map)
    return int_map

#%% Create a binary mask to Highlight the features
from skimage import morphology
def create_mask(map2d, value_threshold, min_size, area_threshold):
    '''
    Create a binary mask based on the 2D map
    :param map2d (np.ndarray): the 2D map
    :param min_size (int): the minimum size of the object
    :param area_threshold (int): the area threshold
    :return: mask (np.ndarray): the binary mask
    '''
    # convert the map with float values to grayscale image
    map2d_gray = np.rint((map2d/map2d.max())*255)
    # create a binary mask (for example, fiducial marker area is 0, the rest is 1)
    mask = morphology.remove_small_holes(morphology.remove_small_objects(map2d_gray > value_threshold, min_size=min_size),
                                         area_threshold=area_threshold)
    # convert the mask to uint8
    mask = mask.astype(np.uint8)
    return mask

#%% Image registration using ANTs with optional parameters
def ants_registration(fixed_img, moving_img,type_of_transform,random_seed=None,interpolator='nearestNeighbor',outprefix=None,
                      savefig=False,figname=None,savepath=None):
    """
    Image registration using ANTs
    :param fixed_img (np.ndarray): the fixed image
    :param moving_img (np.ndarray): the moving image
    :param type_of_transform (str): the type of the transformation supported by ANTs
    :param random_seed (int)(optional): the random seed, used to improve the reproducibility of the registration
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :return: transform (dict): the transformation matrix
        warped_moving (np.ndarray): the registered moving image
    """
    fixed = ants.from_numpy(fixed_img)
    moving = ants.from_numpy(moving_img)

    transform = ants.registration(fixed=fixed, moving=moving, type_of_transform=type_of_transform,random_seed=random_seed,outprefix=outprefix)
    warped_moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transform['fwdtransforms'], interpolator=interpolator).numpy()
    #check the results of the registration
    fig, ax = plt.subplots()
    ax.imshow(fixed_img, cmap='viridis')
    ax.imshow(warped_moving, alpha=0.5, cmap='magma')
    # add a text box to show the random seed
    if random_seed is not None:
        ax.text(0.5, 0.95, 'Random seed: {}'.format(random_seed), horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, fontsize=12, color='white')
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
    plt.show()
    return transform, warped_moving

#%% Checking the registration results
def check_registration(fixed_img, warped_moving):
    """
    Check the registration results
    :param fixed_img (np.ndarray): the fixed image
    :param warped_moving (np.ndarray): the registered moving image
    :return: None
    """
    fig, ax = plt.subplots()
    ax.imshow(fixed_img, cmap='viridis')
    ax.imshow(warped_moving, alpha=0.7, cmap='magma')
    plt.tight_layout()
    plt.show()

#%% Apply the transform to the entire hyperspectral image dataset
def transform2hsi(original_fixed_data, original_moving_data, transform, interpolator='nearestNeighbor'):
    """
    Apply the transformation to the entire hyperspectral image dataset
    :param original_fixed_data (LumiSpectrum): the original hyperspectral image dataset
    :param original_moving_data (LumiSpectrum): the original hyperspectral image dataset
    :param transform (dict): the transformation matrix
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :return: warped_data (np.ndarray): the registered hyperspectral image dataset
    """
    warped_data = np.zeros(original_fixed_data.data.shape)
    for i in range(original_fixed_data.data.shape[2]):
        fixed_data = ants.from_numpy(original_fixed_data.data[:,:,i])
        moving_data = ants.from_numpy(original_moving_data.data[:,:,i])
        warped_data[:,:,i] = ants.apply_transforms(fixed=fixed_data, moving=moving_data, transformlist=transform['fwdtransforms'],interpolator=interpolator).numpy()
    return warped_data
#%% Apply the transform to a 2d map/image
def transform2map(fixed_data, moving_data, transform, interpolator='nearestNeighbor'):
    """
    Apply the transformation to a 2d map/image
    :param fixed_data (np.ndarray): the fixed image
    :param moving_data (np.ndarray): the moving image
    :param transform (dict): the transformation matrix
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :return: warped_data (np.ndarray): the registered image
    """
    fixed = ants.from_numpy(fixed_data)
    moving = ants.from_numpy(moving_data)
    warped_map = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transform['fwdtransforms'], interpolator=interpolator).numpy()
    return warped_map

#%% Get the explained variance ratio plot (scree plot)
def plot_evr(data, n_components):
    """
    Plot the explained variance ratio by PCA
    :param data (LumiSpectrum): hyperspectral data
    :param n_components: the number of principal components shown in the plot
    :return: None
    """
    evr = data.get_explained_variance_ratio()
    plt.plot(evr.data[:n_components], 'ro', markersize=5)
    plt.xlabel('Principal component index',fontsize=12,labelpad=10)
    plt.ylabel('Explained variance ratio',fontsize=12,labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale('log')
    plt.tick_params(which='both', direction='in',right=True, top=True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tight_layout()
    plt.show()

#%% Plot the PC spectra
def PC_spc(data,component_index, data_type,save_path=None):
    """
    Plot the spectrum of the individual principal component
    :param data (LumiSpectrum): hyperspectral data
    :param component_index (int): the index of the principal component starting from 0 (Note the index given by Hyperspy is starting from 0, not 1)
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param save_path (str)(optional): the path to save the figure
    """
    PCs = data.get_decomposition_factors()
    Spectr = PCs.axes_manager[1].axis
    Intens = PCs.data[component_index,:]
    fig, ax = plt.subplots()
    ax.plot(Spectr, Intens)
    if data_type == 'PL':
        xlabel = 'Wavelength (nm)'
        ylabel = 'PL intensity (counts)'
        def lambda2energy(Spectr):
            return 1239.8 / Spectr
        def energy2lambda(Spectr):
            return 1239.8 / Spectr
        secx=ax.secondary_xaxis('top', functions=(lambda2energy, energy2lambda))
        secx.set_xlabel('Energy (eV)', fontsize=12, labelpad=10)
        secx.tick_params(which='both', direction='in', right=True, top=True)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    elif data_type == 'Raman':
        xlabel = 'Raman shift (cm$^{-1}$)'
        ylabel = 'Raman intensity (counts)'
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_title('{} spectrum based on the {}. PC'.format(data_type,component_index+1)) # the index shown in the title is starting from 1
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

#%% Plot the loading map of the given PC
def loading_map(dataPCA, component_index,save_path=None):
    '''
    Plot the loading map of the given PC
    :param dataPCA: the data after PCA decomposition
    :param component_index: the index of the principal component starting from 1
    :return:
    '''
    loading_maps = dataPCA.get_decomposition_loadings()
    loading_map = loading_maps.data[component_index-1,:,:]
    hist, bins = np.histogram(loading_map.flatten(), bins=500)
    cum_counts = np.cumsum(hist)
    tot_counts = np.sum(hist)
    tot_counts_5 = tot_counts * 0.05
    tot_counts_95 = tot_counts * 0.95
    bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]

    scalebar = ScaleBar(dataPCA.axes_manager[0].scale, "um", length_fraction=0.13, location='lower right',
                        box_color=None, color='white', frameon=False, width_fraction=0.02, font_properties={'size': 12,
                                                                                                        'weight': 'bold',
                                                                                                        'math_fontfamily': 'dejavusans'})
    fig, ax = plt.subplots()
    cmap = ax.imshow(loading_map, vmin=bin_edges[0], vmax=bin_edges[-1])
    ax.set_axis_off()
    ax.add_artist(scalebar)
    ax.set_title('Loading map of the {}. PC'.format(component_index))
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Loading value')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

#%% Reconstruct the data using given number of PCs or a certain PC
def rec_data(data, num_PCs, single_PC=None):
    """
    Reconstruct the data using the given number of PCs or a certain PC
    :param data (LumiSpectrum): hyperspectral data
    :param num_PCs (int)(optional): the number of principal components or the index of the certain component starting from 0
    :param single_PC (tuple)(optional): if True, the single component is used.
    :return: rec_data (np.ndarray): the reconstructed data
    """
    if single_PC is True:
        rec_data = data.get_decomposition_model(components=[num_PCs])
    else:
        rec_data = data.get_decomposition_model(components=num_PCs)
    return rec_data
#%% Define gaussian fitting function for the triple Gaussian
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
def triple_gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3):
    return (amp1 * np.exp(-(x - cen1)**2 / (2 * wid1**2)) +
            amp2 * np.exp(-(x - cen2)**2 / (2 * wid2**2)) +
            amp3 * np.exp(-(x - cen3)**2 / (2 * wid3**2)))

# Function to integrate a Gaussian function
def gaussian_integral(amplitude, width):
    return amplitude * np.sqrt(2 * np.pi) * width
#%% tripple gaussian fitting to extract parameters and errors
def gaussian_fit(data, func_name='triple_gaussian',initial_guesses=None,bounds=(-np.inf,np.inf)):
    """
    Fit the PL spectrum of a single pixel with the triple Gaussian peaks
    :param data (LumiSpectrum): the original (or reconstructed) hyperspectral data read by hyperspy
    :param processed_data (np.ndarray): the warped hyperspectral data
    :param initial_guesses (list): the initial guesses for the Gaussian parameters
    :return: params (np.ndarray): the Gaussian parameters,
                errors (np.ndarray): the errors of the Gaussian parameters
    """
    if func_name == 'triple_gaussian':
        func = triple_gaussian
    fit_params = np.zeros((data.data.shape[0], data.data.shape[1], 9))
    fit_errors = np.zeros((data.data.shape[0], data.data.shape[1], 9))
    wavelengths = data.axes_manager[2].axis
    for i in range(data.data.shape[0]):
        for j in range(data.data.shape[1]):
            Spectra = data.data[i, j, :]
            try:
                # Fit the spectrum with the triple Gaussian function
                popt, cov = curve_fit(func, wavelengths, Spectra, p0=initial_guesses,bounds=bounds)
                errors = np.sqrt(np.diag(cov))
                fit_params[i, j, :] = popt
                fit_errors[i, j, :] = errors
            except RuntimeError:
                # If the fit fails, set the Gaussian parameters and errors to zero
                print('The fit failed at pixel [{}, {}]'.format(i, j))
                fit_params[i, j, :] = 0
                fit_errors[i, j, :] = 0
    return fit_params, fit_errors
#%% Plot spectra with the Gaussian fitting and the residuals at the given pixel
def plot_gaussian_fit(data, params, func_name='triple_gaussian', px_yx=(0,0),save_path=None):
    """
    Plot the spectrum with the triple Gaussian fitting and the residuals at the given pixel
    :param original_data (LumiSpectrum): the original hyperspectral data
    :param params (np.ndarray): the Gaussian parameters
    :param px_xy (tuple): the coordinates of the pixel in the format (y, x)
    :param save_path (str)(optional): the path to save the figure
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    wavelengths = data.axes_manager[2].axis
    spectrum = data.data[px_yx[0], px_yx[1], :]
    params_pixel = params[px_yx[0], px_yx[1], :]
    ax[0].scatter(wavelengths, spectrum, label='Data')
    if func_name == 'triple_gaussian':
        func = triple_gaussian
        fit1 = gaussian(wavelengths, *params_pixel[:3])
        fit2 = gaussian(wavelengths, *params_pixel[3:6])
        fit3 = gaussian(wavelengths, *params_pixel[6:])
        ax[0].plot(wavelengths, fit1, label='Gaussian 1',color='y')
        ax[0].plot(wavelengths, fit2, label='Gaussian 2',color='g')
        ax[0].plot(wavelengths, fit3, label='Gaussian 3',color='c')
    elif func_name == 'gaussian':
        func = gaussian
    fit = func(wavelengths, *params_pixel)
    residuals = spectrum - fit
    ax[0].plot(wavelengths, fit, label='Fit',color='r')
    ax[1].scatter(wavelengths, residuals, label='Residuals')
    ax[0].set_xlabel('Wavelength (nm)',fontsize=12,labelpad=10)
    ax[0].set_ylabel('PL intensity (counts)',fontsize=12,labelpad=10)
    ax[1].set_xlabel('Wavelength (nm)',fontsize=12,labelpad=10)
    ax[1].set_ylabel('Data - Fit',fontsize=12,labelpad=10)
    ax[0].tick_params(which='both', direction='in', right=True, top=True)
    ax[1].tick_params(which='both', direction='in', right=True, top=True)
    ax[0].legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
    return None
#%% Functions used to get the integrated intensity of the Gaussian peaks over the entire spectral range
# Step1: Fit each pixel with multiple Gaussian peaks to extract the integrated intensity of the individual peaks
from scipy.optimize import curve_fit
def intint_gaussian(original_data, initial_guesses=None, sum_int_threshold=0,bounds=(-np.inf,np.inf)):
    """
    Fit the PL spectrum of a single pixel with the triple Gaussian peaks and extract the integrated intensity of the individual peaks
    :param original_data (LumiSpectrum): the original hyperspectral data read by hyperspy
    :param initial_guesses (list): the initial guesses for the Gaussian parameters
    :param sum_int_threshold (int): the threshold for the sum of the intensity of the spectrum
    :param bounds (tuple): the bounds for the Gaussian parameters
    :return: intint_gaussian1 (np.ndarray): the integrated intensity map of the 1st Gaussian peak,
                intint_gaussian2 (np.ndarray): the integrated intensity map of the 2nd Gaussian peak,
                intint_gaussian3 (np.ndarray): the integrated intensity map of the 3rd Gaussian peak
    """
    wavelengths = original_data.axes_manager[2].axis
    # Initialize intensity maps for each Gaussian
    map_shape = original_data.data.shape
    intint_gaussian1 = np.zeros(map_shape[:2])
    intint_gaussian2 = np.zeros(map_shape[:2])
    intint_gaussian3 = np.zeros(map_shape[:2])

    # Loop through each pixel in the spatial dimensions
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            spectrum = original_data.data[i, j, :]
            # mask the fiducial marker area, the intensity sum of the pixel is below threshold (according to the rough PL integrated intensity map)
            if spectrum.sum() < sum_int_threshold:
                intint_gaussian1[i, j] = 0
                intint_gaussian2[i, j] = 0
                intint_gaussian3[i, j] = 0
                continue
            try:
                # Fit the spectrum with the triple Gaussian function
                params, _ = curve_fit(triple_gaussian, wavelengths, spectrum, p0=initial_guesses,bounds=bounds)

                # Extract Gaussian parameters and calculate the integrated intensities
                amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3 = params
                intint_gaussian1[i, j] = gaussian_integral(amp1, wid1)
                intint_gaussian2[i, j] = gaussian_integral(amp2, wid2)
                intint_gaussian3[i, j] = gaussian_integral(amp3, wid3)
            except RuntimeError:
                # If the fit fails, set the integrated intensities to zero
                print('The fit failed at pixel ({}, {})'.format(i, j))
                intint_gaussian1[i, j] = 0
                intint_gaussian2[i, j] = 0
                intint_gaussian3[i, j] = 0

    return intint_gaussian1, intint_gaussian2, intint_gaussian3

#%% Step2: Plot the integrated intensity map of the individual Gaussian peak over the entire spectral range
def plot_intint_gaussian(original_data,intint_gaussian,cbar_label='PL integrated intensity (counts)',ROI=None,save_path=None):
    """
    Plot the integrated intensity maps of the individual Gaussian peaks over the entire spectral range
    :param original_data (LumiSpectrum): the original hyperspectral data used to provide the scale information
    :param intint_gaussian (np.ndarray): the integrated intensity map of the individual Gaussian peaks
    :param cbar_label (str): the label of the colorbar
    :param ROI (tuple)(optional): the region of interest (ROI) in the format (x, y, width, height)
    :param save_path (str)(optional): the path to save the figure
    :return: None
    """
    # integrate a single Gaussian peak
    # use histogram to avoid the bad pixel in order to get a better ratio map
    hist, bins = np.histogram(intint_gaussian.flatten(), bins=500)
    cum_counts = np.cumsum(hist)
    tot_counts = np.sum(hist)
    tot_counts_5 = tot_counts * 0.05
    tot_counts_95 = tot_counts * 0.95
    bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]
    fig, ax = plt.subplots()
    if ROI is not None:
        intint_gaussian_ROI = intint_gaussian[ROI[1]:ROI[1]+ROI[3],ROI[0]:ROI[0]+ROI[2]]
        cmap = ax.imshow(intint_gaussian_ROI, vmin=bin_edges[0], vmax=bin_edges[-1])
    else:
        cmap = ax.imshow(intint_gaussian, vmin=bin_edges[0], vmax=bin_edges[-1])
    ax.set_axis_off()
    scalebar = ScaleBar(original_data.axes_manager[0].scale, "um", length_fraction=0.13, location='lower right',
                        box_color=None, color='white', frameon=False, width_fraction=0.02, font_properties={'size': 12,
                                                                                                        'weight': 'bold',
                                                                                                        'math_fontfamily': 'dejavusans'})
    ax.add_artist(scalebar)
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

#%% Plot the centre of mass map
# Calculate and plot the COM of the PL spectrum for all pixels using vectorized operations
def plot_com_map(original_data, processed_data=None, lambda1=None, lambda2=None, params_ROI=None,data_type='PL',save_path=None):
    """
    Plot the centre of mass (COM) map of the PL spectrum for all pixels
    :param original_data (LumiSpectrum): the original/reconstructed hyperspectral data
    :param processed_data (np.ndarray): the warped hyperspectral data
    :param lambda1 (float)(optional): the lower limit of the wavelength range for the COM calculation
    :param lambda2 (float)(optional): the upper limit of the wavelength range for the COM calculation
    :param params_ROI (tuple)(optional): the region of interest (ROI) in the format (x, y, width, height)
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param save_path (str)(optional): the path to save the figure
    :return: com_map (np.ndarray): the COM map
    """
    index1 = 0
    index2 = original_data.data.shape[2]-1
    # lambda1 and lambda2 are the wavelength range for the COM calculation
    if lambda1 is not None:
        index1 = abs(original_data.axes_manager[2].axis-lambda1).argmin()
    if lambda2 is not None:
        index2 = abs(original_data.axes_manager[2].axis-lambda2).argmin()
    if processed_data is not None:
        intensities = processed_data[:,:,index1:index2]
    else:
        intensities = original_data.data[:,:,index1:index2]
    wavelength = original_data.axes_manager[2].axis[index1:index2]
    # Calculating COM for all pixels using vectorized operations
    numerator = np.sum(wavelength * intensities, axis=2)
    denominator = np.sum(intensities, axis=2)
    com_map = numerator / denominator
    if data_type == 'PL':
        #convert wavelength to energy
        #com_map = 1239.8 / com_map
        cbarlabel = 'PL center of mass energy (nm)'
    elif data_type == 'Raman':
        cbarlabel = 'Raman center of mass energy (cm$^{-1}$)'
    # check if the is NaN and replace it with zero
    com_map = np.nan_to_num(com_map)
    # adjust the color scale using the histogram
    hist, bins = np.histogram(com_map.flatten(), bins=500)
    cum_counts = np.cumsum(hist)
    tot_counts = np.sum(hist)
    tot_counts_5 = tot_counts * 0.05
    tot_counts_95 = tot_counts * 0.95
    bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]
    fig, ax = plt.subplots()
    if params_ROI is not None:
        com_map_ROI = com_map[params_ROI[1]:params_ROI[1]+params_ROI[3],params_ROI[0]:params_ROI[0]+params_ROI[2]]
        hist, bins = np.histogram(com_map_ROI.flatten(), bins=500)
        cum_counts = np.cumsum(hist)
        tot_counts = np.sum(hist)
        tot_counts_5 = tot_counts * 0.05
        tot_counts_95 = tot_counts * 0.95
        bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]
        cmap = ax.imshow(com_map_ROI,vmin=bin_edges[0], vmax=bin_edges[-1])
    else:
        cmap = ax.imshow(com_map,vmin=bin_edges[0], vmax=bin_edges[-1])
    ax.set_axis_off()
    scalebar = ScaleBar(original_data.axes_manager[0].scale, "um", length_fraction=0.13, location='lower right',
                        box_color=None, color='white', frameon=False, width_fraction=0.02, font_properties={'size': 12,
                                                                                                        'weight': 'bold',
                                                                                                        'math_fontfamily': 'dejavusans'})
    ax.add_artist(scalebar)
    cbar = fig.colorbar(cmap, ax=ax, format='%.1f')
    cbar.set_label(cbarlabel)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
    return com_map

#%% Extract the coordinates of the special values such as the maximum, minimum, and the median
def coord_extract(map_data, value='max'):
    '''
    Extract the coordinates of the maxima, minima or median in the map
    :param map_data (np.ndarray): the 2D map data
    :param value: 'max' by default, which refers to maximum, 'min' for minima, 'median' for median;
    :return: coord (tuple): (Y, X) the coordinate of the point with the specified value
    '''
    coord = np.unravel_index(np.argmax(map_data, axis=None), map_data.shape)
    if value == 'min':
        coord = np.unravel_index(np.argmin(map_data, axis=None), map_data.shape)
    elif value == 'median':
        coord = np.unravel_index(np.argsort(map_data, axis=None)[map_data.size // 2], map_data.shape)
    return coord

#%% Mark points of interest on the map
def point_marker(map_data,original_data, YX,cbarlabel='Intensity (counts)',save_path=None):
    """
    Mark the points of interest on the map
    :param map_data (np.ndarray): the dataset of map
    :param original_data (LumiSpectrum): the original data read by hyperspy used to provide the scale information
    :param YX (list): the list of the coordinates of the points of interest (Y,X)
    :param cbarlabel (str): the label of the colorbar
    :param save_path (str)(optional): the path to save the figure
    """
    # use histogram to avoid the bad pixel in order to get a better ratio map
    hist, bins = np.histogram(map_data.flatten(), bins=500)
    # find the bin-edges for the range from the maximal counts of 5% to the maximal counts of 95% in the histogram
    cum_counts = np.cumsum(hist)
    tot_counts = np.sum(hist)
    tot_counts_5 = tot_counts * 0.05
    tot_counts_95 = tot_counts * 0.95
    bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]
    # define scalebar
    scalebar = ScaleBar(original_data.axes_manager[0].scale, "um", length_fraction=0.13, location='lower right',
                        box_color=None, color='white', frameon=False, width_fraction=0.02, font_properties={'size': 12,
                                                                                                        'weight': 'bold',
                                                                                                        'math_fontfamily': 'dejavusans'})
    fig,ax = plt.subplots()
    cmap=ax.imshow(map_data,vmin=bin_edges[0], vmax=bin_edges[-1])
    ax.set_axis_off()
    ax.add_artist(scalebar)
    colors = ['m', 'k', 'b', 'r', 'c', 'g']
    for i in range(len(YX)):
        ax.plot(YX[i][1], YX[i][0], 'o', mfc='none', mec=colors[i],mew=3, markersize=15)
    cbar=fig.colorbar(cmap, ax=ax, format='%.1f')
    cbar.set_label(cbarlabel, fontsize=12, labelpad=10)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()


#%% Extract the spectrum at the points of interest and plot them on the same figure
def Spectrum_extracted(original_data, YX, data_type,processed_data=None,x_lim=None,y_lim=None,spc_labels=None,save_path=None):
    """
    Plot the spectrum at the points of interest on the same figure
    :param original_data (LumiSpectrum): hyperspectral data read by Hyperspy used to provide the spectral axis information
    :param YX (list): the list of the coordinates of the points of interest (Y,X)
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param processed_data (np.ndarray)(optional): the registered (and reconstructed) data after data preprocessing
    :param x_lim (list)(optional): the x limit of the plot
    :param y_lim (list)(optional): the y limit of the plot
    :return: fig (matplotlib.figure.Figure): the figure object,
                ax (matplotlib.axes._axes.Axes): the axes object
    """
    if spc_labels is None:
        # create a list of None with the same length as YX
        spc_labels = [None] * len(YX)
    else:
        spc_labels = spc_labels
    colors = ['m', 'k', 'b', 'r', 'c', 'g'] # set the color map, here we use the RdPu colormap
    fig, ax = plt.subplots()
    for i in range(len(YX)):
        x = YX[i][1]
        y = YX[i][0]
        Spectr = original_data.axes_manager[2].axis
        if processed_data is not None:
            intensity = processed_data[y, x, :]
        else:
            intensity = original_data.data[y, x, :]
        ax.plot(Spectr, intensity, color=colors[i], label=spc_labels[i])
    if data_type == 'PL':
        ax.set_xlabel('Wavelength (nm)', fontsize=12, labelpad=10)
        secx = ax.secondary_xaxis('top', functions=(lambda Spectr: 1239.8 / Spectr, lambda Spectr: 1239.8 / Spectr))
        secx.set_xlabel('Energy (eV)', fontsize=12, labelpad=10)
        secx.tick_params(which='both', direction='in', right=True, top=True)
        plt.gca().xaxis.set_major_locator(MultipleLocator(50))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(which='both', direction='in', right=True, top=False)
    elif data_type == 'Raman':
        ax.set_xlabel('Raman shift (cm$^{-1}$)', fontsize=12, labelpad=10)
        ax.tick_params(which='both', direction='in', right=True, top=True)
        plt.gca().xaxis.set_major_locator(MultipleLocator(200))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(which='both', direction='in', right=True, top=True)
    if x_lim is not None:
        ax.set_xlim(x_lim)
        plt.gca().xaxis.set_major_locator(MultipleLocator(x_lim[1]/10))
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel('{} intensity (counts)'.format(data_type), fontsize=12, labelpad=10)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()


#%% Plot an average spectrum
def avg_spectrum(data, data_type,params_ROI=None,save_path=None):
    """
    Plot the average spectrum over the whole map
    :param data (LumiSpectrum): hyperspectral data
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param params_ROI (tuple)(optional): the region of interest (ROI) in the format (x, y, width, height)
    :param save_path (str)(optional): the path to save the figure
    :return:
        avg_intens (np.ndarray): the average intensity spectrum
    """
    Spectr = data.axes_manager[2].axis
    if params_ROI is not None:
        #plot the average spectrum over a region of interest (for example, (28,20,31,31) which means that
        # the starting point is xy:[28,10] and the width in x and y axes are 31)
        avg_intens = data.data[params_ROI[1]:params_ROI[1]+params_ROI[3],params_ROI[0]:params_ROI[0]+params_ROI[2],:].mean(axis=(0,1))
    else:
        avg_intens = data.data.mean(axis=(0,1))
    fig, ax = plt.subplots()
    ax.plot(Spectr, avg_intens)
    if data_type == 'PL':
        x_label = 'Wavelength (nm)'
        def lambda2energy(Spectr):
            return 1239.8 / Spectr

        def energy2lambda(Spectr):
            return 1239.8 / Spectr
        secx=ax.secondary_xaxis('top', functions=(lambda2energy, energy2lambda))
        secx.set_xlabel('Energy (eV)', fontsize=12, labelpad=10)
        secx.tick_params(which='both', direction='in', right=True, top=True)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    elif data_type == 'Raman':
        x_label = 'Raman shift (cm$^{-1}$)'
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel('{} intensity (counts)'.format(data_type), fontsize=12, labelpad=10)
    ax.tick_params(which='both', direction='in', right=True, top=False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
    return avg_intens

#%% Compare the spectra on the same figure, for example, the average spectrum before & after illumination
def spectra_compare(data1, data2, data_type,original_data1, original_data2=None,lambda_range=None,save_path=None):
    """
    Plot two spectra on the same figure
    :param data1 (np.ndarray): spectrum 1
    :param data2 (np.ndarray): spectrum 2
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param original_data1 (LumiSpectrum): the original data read by Hyperspy providing spectral info for data1
    :param original_data2 (LumiSpectrum)(optional): the original data read by Hyperspy providing spectral info for data2,
            if none, the spectral axis of data2 is the same as that of data1
    :param lambda_range (tuple)(optional): the range of the wavelength/wavenumber for the plot
    :param save_path (str)(optional): the path to save the figure
    :return:
    """
    fig, ax = plt.subplots()
    if lambda_range is not None:
        index1 = abs(original_data1.axes_manager[2].axis-lambda_range[0]).argmin()
        index2 = abs(original_data1.axes_manager[2].axis-lambda_range[1]).argmin()
        data1 = data1[index1:index2]
        data2 = data2[index1:index2]
    else:
        index1 = 0
        index2 = original_data1.data.shape[2]-1
    ax.plot(original_data1.axes_manager[2].axis[index1:index2], data1[index1:index2], label='Before')
    if original_data2 is not None:
        ax.plot(original_data2.axes_manager[2].axis[index1:index2], data2[index1:index2], label='After')
    else:
        ax.plot(original_data1.axes_manager[2].axis[index1:index2], data2[index1:index2], label='After')
    if data_type == 'PL':
        x_label = 'Wavelength (nm)'
        def lambda2energy(Spectr):
            return 1239.8 / Spectr

        def energy2lambda(Spectr):
            return 1239.8 / Spectr
        secx = ax.secondary_xaxis('top', functions=(lambda2energy, energy2lambda))
        secx.set_xlabel('Energy (eV)', fontsize=12, labelpad=10)
        secx.tick_params(which='both', direction='in', right=True, top=True)
        if lambda_range is not None:
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    elif data_type == 'Raman':
        x_label = 'Raman shift (cm$^{-1}$)'
        if lambda_range is not None:
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel('{} intensity (counts)'.format(data_type), fontsize=12, labelpad=10)
    ax.tick_params(which='both', direction='in', right=True, top=True)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
#%% Find local maxima in the spectrum based on scipy.signal.find_peaks
import scipy.signal as signal

def find_maxima(original_data, spectrum_data, data_type, prominence=None,height=None,save_path=None):
    """
    Find the maxima in the average spectrum
    :param original_data (LumiSpectrum): the original data read by Hyperspy providing spectral information
    :param spectrum_data (np.ndarray): the spectrum
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param prominence (int or float): the prominence of the peaks
    :param height (int or float): the height of the peaks
    :param save_path (str)(optional): the path to save the figure
    :return: peaks (np.ndarray): the indices of the maxima in the spectrum
    """
    peaks, _ = signal.find_peaks(spectrum_data, prominence=prominence,height=height)
    pps = np.array(original_data.axes_manager[2].axis[peaks])  # peak positions
    pint = np.array(spectrum_data[peaks])  # peak intensity
    plt.plot(original_data.axes_manager[2].axis, spectrum_data)
    plt.plot(pps, pint, 'x', label='Maxima')
    for p in range(len(pps)):
        plt.text(pps[p], pint[p]+2, np.round(pps[p]), fontsize=10, rotation='vertical', va='bottom', ha='center')
    if data_type == 'PL':
        x_label = 'Wavelength (nm)'
        def lambda2energy(Spectr):
            return 1239.8 / Spectr

        def energy2lambda(Spectr):
            return 1239.8 / Spectr
        secx = plt.gca().secondary_xaxis('top', functions=(lambda2energy, energy2lambda))
        secx.set_xlabel('Energy (eV)', fontsize=12, labelpad=10)
        secx.tick_params(which='both', direction='in', right=True, top=True)
        plt.gca().xaxis.set_major_locator(MultipleLocator(50))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    elif data_type == 'Raman':
        x_label = 'Raman shift (cm$^{-1}$)'
        plt.gca().xaxis.set_major_locator(MultipleLocator(500))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.xlabel(x_label, fontsize=12, labelpad=10)
    plt.ylabel('{} intensity (counts)'.format(data_type), fontsize=12, labelpad=10)
    plt.tick_params(which='both', direction='in', right=True, top=True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
    return peaks

#%% Plot histogram(s) of the 2d map data
import scipy.stats as stats
def plot_hist(dataMap, labels=None, spread=True, bins=100, bins_range=None, x_label='Intensity (counts)', save_path=None):
    """
    Plot histogram(s) of the 2D map data
    :param dataMap (np.ndarray or list): the 2D map data or a list of 2D map data
    :param labels (str or list)(optional): the label(s) of the histogram(s)
    :param spread (bool): if True, the histogram(s) will be spread out
    :param bins (int): the number of bins
    :param bins_range (tuple)(optional): the range of the bins
    :param x_label (str): the label of the x-axis
    :param save_path (str)(optional): the path to save the figure
    """
    # check if the dataMap is a list
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    if isinstance(dataMap,list):
        for i in range(len(dataMap)):
            mean, std = stats.norm.fit(dataMap[i].flatten())
            if bins_range is not None:
                Range = bins_range
            else:
                Range = (dataMap[i].flatten().min(), dataMap[i].flatten().max())
            if spread:
                ax.hist(dataMap[i].flatten(), bins=bins, range=Range, color=colors[i], alpha=0.5, label=labels[i]+': mean={:.1f}, std={:.1f}'.format(mean, std))
                # plot the corresponding normal distribution
                x = np.linspace(Range[0], Range[1], bins)
                y = stats.norm.pdf(x, mean, std)
                ax.plot(x, y, color=colors[i])
            else:
                ax.hist(dataMap[i].flatten(), bins=bins, range=Range, color=colors[i], alpha=0.5, label=labels[i])
    else:
        mean, std = stats.norm.fit(dataMap.flatten())
        if bins_range is not None:
            Range = bins_range
        else:
            Range = (dataMap.flatten().min(), dataMap.flatten().max())
        if spread:
            ax.hist(dataMap.flatten(), bins=bins, range=Range, color=colors[0], alpha=0.5, label=labels+': mean={:.1f}, std={:.1f}'.format(mean, std))
            # plot the corresponding normal distribution
            x = np.linspace(Range[0], Range[1], bins)
            y = stats.norm.pdf(x, mean, std)
            ax.plot(x, y, color=colors[0])
        else:
            ax.hist(dataMap.flatten(), bins=bins, range=Range, color=colors[0], alpha=0.5, label=labels)
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency', fontsize=12, labelpad=10)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()