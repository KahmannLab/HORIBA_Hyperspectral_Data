import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import copy
#%% extract the xml file paths
def xml_paths(folder_path, endswith='.xml', IDs=None):
    """
    Extract the sif file paths from the folder, based on the given file extension or keywords
    :param folder_path (str): the path to the folder
    :param endswith (str): the file extension to filter the files, default is '.xml'
    :param IDs (list)(optional): the list of the IDs to filter the files, default is None
    :return: sif_paths (list): the list of the sif file paths
    """
    xml_paths = []
    if IDs is not None:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(endswith) and any(ID in file for ID in IDs):
                    xml_paths.append(os.path.join(root, file))
        return xml_paths

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(endswith):
                xml_paths.append(os.path.join(root, file))
    return xml_paths
#%% Remove the values below 0 from the raw data
def remove_negative(data):
    """
    Modify the raw data by removing the values below 0
    :param  data (LumiSpectrum): the original hyperspectral data
    :return: data (LumiSpectrum): the modified hyperspectral data
    """
    data.data[data.data < 0] = 0
    return data
#%% Load the hyperspectral data in an .xml file exported(or saved) from LabSpec(HORIBA) software
def load_xml(file_path,remove_spikes=False,threshold='auto', remove_negatives=False,plot=False):
    '''
    Load the hyperspectral data in an .xml file
    :param file_path (str): the path of the hyperspectral data file
    :param remove_spikes (bool): whether to remove the spikes in the data, default is False
    :param threshold_factor (int): the factor to determine the threshold for removing spikes, default is 5
    :param remove_negatives (bool): whether to remove the negative values in the data, default is False
    :return:
    '''
    data = hs.load(file_path, reader='JobinYvon')
    if remove_spikes:
        data.spikes_removal_tool(threshold=threshold,interactive=False)
    if remove_negatives:
        data = remove_negative(data)
    if plot:
        # check if the filename contains PL or Raman
        if 'PL' in os.path.basename(file_path):
            xlabel = 'Wavelength / nm'
            ylabel = 'PL intensity / a.u.'
        elif 'Raman' in os.path.basename(file_path):
            xlabel = 'Raman shift / cm$^{-1}$'
            ylabel = 'Raman intensity / a.u.'
        else:
            xlabel = 'Wavelength / nm'
            ylabel = 'Intensity / a.u.'
        visual_data(data, xlabel=xlabel, ylabel=ylabel)
    return data
#%% visualize the hyperspectral data
def visual_data(data, xlabel='Wavelength / nm', ylabel='PL intensity / a.u.',
                fontsize=12,labelpad=10, labelsize=12,
                savefig=False, figname=None, savepath=None):
    """
    Plot all spectra of hyperspectral data in a single plot
    :param data (np.ndarray): the 2D dataset
    :param xlabel (str): the label of the x-axis
    :param ylabel (str): the label of the y-axis
    :param savefig (bool): whether to save the figure
    :param figname (str): the name of the figure
    :param savepath (str): the path to save the figure
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(data.data.shape[0]):
        for j in range(data.data.shape[1]):
            ax.plot(data.axes_manager[2].axis, data.data[i, j, :])
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
    ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
    plt.show()
#%% extract mapping data from datacube
# get integrated intensity of the hyperspectral data
def get_intint(data, spectral_range=None, processed_data=None):
    """
    Get the integrated intensity of the hyperspectral data over the given wavelength or wavenumber range
    :param data (LumiSpectrum): hyperspectral data
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param processed_data (np.ndarray)(optional): the registered data
    :return: intint (np.ndarray): the integrated intensity map
    """
    Spectr = data.axes_manager[2].axis
    if spectral_range is not None:
        index1 = abs(Spectr - spectral_range[0]).argmin()
        index2 = abs(Spectr - spectral_range[1]).argmin()
    else:
        index1 = 0
        index2 = data.data.shape[2]-1
    if processed_data is not None:
        intint = processed_data[:,:,index1:index2].sum(axis=2)
    else:
        intint = data.data[:,:,index1:index2].sum(axis=2)
    return intint
# get maximum intensity of each pixel
def get_maxint(data, spectral_range=None, processed_data=None):
    """
    Get the maximum intensity of each pixel in the hyperspectral data over the given wavelength or wavenumber range
    :param data (LumiSpectrum): hyperspectral data
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param processed_data (np.ndarray)(optional): the registered data
    :return: maxint (np.ndarray): the maximum intensity map
    """
    Spectr = data.axes_manager[2].axis
    if spectral_range is not None:
        index1 = abs(Spectr - spectral_range[0]).argmin()
        index2 = abs(Spectr - spectral_range[1]).argmin()
    else:
        index1 = 0
        index2 = data.data.shape[2]-1
    if processed_data is not None:
        maxint = processed_data[:,:,index1:index2].max(axis=2)
    else:
        maxint = data.data[:,:,index1:index2].max(axis=2)
    return maxint
# get the intensity at the given wavelength or wavenumber
def get_int(data, wavelength, processed_data=None):
    """
    Get the intensity at the given wavelength or wavenumber
    :param data (LumiSpectrum): hyperspectral data
    :param wavelength (float): the wavelength or wavenumber
    :param processed_data (np.ndarray)(optional): the registered data
    :return: int_map (np.ndarray): the intensity map at the given wavelength or wavenumber
    """
    Spectr = data.axes_manager[2].axis
    if processed_data is not None:
        int = processed_data[:,:,abs(Spectr-wavelength).argmin()]
    else:
        int = data.data[:,:,abs(Spectr-wavelength).argmin()]
    return int
# get the centre of mass (COM) of the hyperspectral data
def get_com(data, spectral_range=None, processed_data=None):
    """
    Get the centre of mass (COM) of the hyperspectral data over the given wavelength or wavenumber range
    :param data (LumiSpectrum): hyperspectral data
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param processed_data (np.ndarray)(optional): the registered data
    :return: com (np.ndarray): the COM map
    """
    Spectr = data.axes_manager[2].axis
    if spectral_range is not None:
        index1 = abs(Spectr - spectral_range[0]).argmin()
        index2 = abs(Spectr - spectral_range[1]).argmin()
    else:
        index1 = 0
        index2 = data.data.shape[2]-1
    if processed_data is not None:
        com = np.sum(processed_data[:,:,index1:index2] * Spectr[index1:index2], axis=2) / np.sum(processed_data[:,:,index1:index2], axis=2)
    else:
        com = np.sum(data.data[:,:,index1:index2] * Spectr[index1:index2], axis=2) / np.sum(data.data[:,:,index1:index2], axis=2)
    return com
#%% get ratio of peak intensities at given wavelengths or wavenumbers
def get_ratio(data, wavelength1, wavelength2):
    '''
    Get the ratio of the intensity at two given wavelengths or wavenumbers
    :param data:
    :param wavelength1:
    :param wavelength2:
    :return:
    '''
    int1 = get_int(data, wavelength1)
    int2 = get_int(data, wavelength2)
    # Avoid division by zero
    int2[int2 == 0] = 1e-10  # Set a small value to avoid division by zero
    ratio = int1 / int2
    return ratio
#%% get the ideal scalebar length
def get_scalebar_length(data, pixel_to_mum, percent=0.133335):
    """
    Calculate the ideal scalebar length based on the data and pixel size. Best length of a scale bar. 13% of the length of the image.
    :param data: an image data
    :param pixel_to_mum: Pixel size in micrometers.
    :param percent: the percentage of the image length to be used for the scalebar, default is 0.133335 (13%).
    :return:
    len_in_pix (float): the length of the scalebar in pixels
    length (float): the length of the scalebar in micrometers
    width (float): the width of the scalebar in pixels
    """
    ideal_length_scale_bar = data.shape[1] *pixel_to_mum * percent  # 13% of the image length in micrometers

    # Work out how many pixels are required for the scalebar. If scale bar length is > 10 round to the nearest 5.
    if ideal_length_scale_bar > 10:
        n = (ideal_length_scale_bar - 10) / 5
        n = round(n)
        length = int(10 + 5 * n)
        len_in_pix = length / pixel_to_mum

    # Round to the nearest integer if between 1 and 10.
    elif (ideal_length_scale_bar <= 10) & (ideal_length_scale_bar >= 1):
        n = int(round(ideal_length_scale_bar))
        length = n
        len_in_pix = length / pixel_to_mum

    # Round to 1 decimal place if < 1.
    elif ideal_length_scale_bar < 1:
        n = round(ideal_length_scale_bar, 1)
        length = n
        len_in_pix = n / pixel_to_mum

    width = 0.06 * len_in_pix

    return len_in_pix, length, width
#%% adjust colorbar ticks based on the histogram of the data
def adjust_colorbar(data, bins=500, percentiles=(5, 95)):
    """
    Adjust the colorbar ticks based on the histogram of the data.
    :param data: 2D array-like data for which to adjust the colorbar
    :param bins: Number of bins for the histogram
    :param percentiles: Percentiles to use for adjusting the colorbar limits
    :return: vmin, vmax - adjusted colorbar limits
    """
    hist, bin_edges = np.histogram(data.flatten(), bins=bins)
    cum_counts = np.cumsum(hist)
    total_counts = np.sum(hist)

    lower_bound = bin_edges[np.where(cum_counts >= total_counts * percentiles[0] / 100)[0][0]]
    upper_bound = bin_edges[np.where(cum_counts <= total_counts * percentiles[1] / 100)[0][-1]]

    return lower_bound, upper_bound

#%%
def lambda2energy(Spectr):
    return 1239.8 / Spectr

def energy2lambda(Spectr):
    return 1239.8 / Spectr
#%% Plot a combination of integrated intensity map, maximum intensity map, and COM map.
from skimage import exposure
def plot_maps(data,spectral_range=None,data_type='PL',frac_scalebar=0.133335,
              percentile_range=(5, 95),
              savefig=False, figname=None, savepath=None,
              **kwargs):
    """
    Plot a combination of integrated intensity map, maximum intensity map, and COM map.
    :param data (LumiSpectrum): hyperspectral data
    :param spectral_range (list)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param processed_data (np.ndarray)(optional): the registered data
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param frac_scalebar (float): the fraction of the image length to be used for the scalebar, default is 0.133335 (13%)
    :param savefig (bool): whether to save the figure, default is False
    :param figname (str)(optional): the name of the figure, default is None
    :param savepath (str)(optional): the path to save the figure, default is None
    :return:
    """
    intint = get_intint(data, spectral_range=spectral_range, **kwargs)
    maxint = get_maxint(data, spectral_range=spectral_range, **kwargs)
    com = get_com(data, spectral_range=spectral_range, **kwargs)
    # rescale intensity to adjust the contrast
    map_list = [intint, maxint, com]
    v_ranges = []
    for i in range(len(map_list)):
        p1,p2 = np.percentile(map_list[i], q=percentile_range)
        map_list[i] = exposure.rescale_intensity(map_list[i], in_range=(p1, p2))
    """
    # histogram equalization for better contrast
    intint = exposure.equalize_hist(intint, nbins=500)
    maxint = exposure.equalize_hist(maxint, nbins=500)
    com = exposure.equalize_hist(com, nbins=500)
    """
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(intint, data.axes_manager[0].scale, percent=frac_scalebar)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # Plot the integrated intensity map
    cmap1 = ax[0].imshow(map_list[0], cmap='viridis')
    ax[0].set_title('Integrated Intensity Map', fontsize=14)
    ax[0].set_axis_off()
    scalebar1 = AnchoredSizeBar(ax[0].transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                 borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                 fontproperties={'size': 15, 'weight': 'bold'})
    ax[0].add_artist(scalebar1)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar1 = fig.colorbar(cmap1, ax=ax[0], format=fmt)
    cbar1.set_label('{} integrated intensity / a.u.'.format(data_type), fontsize=12, labelpad=10)
    # Plot the maximum intensity map
    cmap2 = ax[1].imshow(map_list[1], cmap='viridis')
    ax[1].set_title('Maximum Intensity Map', fontsize=14)
    ax[1].set_axis_off()
    scalebar2 = AnchoredSizeBar(ax[1].transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                    borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                    fontproperties={'size': 15, 'weight': 'bold'})
    ax[1].add_artist(scalebar2)
    cbar2 = fig.colorbar(cmap2, ax=ax[1])
    cbar2.set_label('{} maximum intensity / a.u.'.format(data_type), fontsize=12, labelpad=10)
    # Plot the COM map
    cmap3 = ax[2].imshow(map_list[2], cmap='viridis')
    ax[2].set_title('Centre of Mass Map', fontsize=14)
    ax[2].set_axis_off()
    scalebar3 = AnchoredSizeBar(ax[2].transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                    borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                    fontproperties={'size': 15, 'weight': 'bold'})
    ax[2].add_artist(scalebar3)
    cbar3 = fig.colorbar(cmap3, ax=ax[2])
    cbar3.set_label('{} COM / nm'.format(data_type), fontsize=12, labelpad=10)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
    plt.show()

    return map_list

#%% Plot an integrated intensity map
def intint_map(data, data_type,
spectral_range=None, processed_data=None,
               frac_scalebar=0.133335,
               savefig=False, figname=None, savefile=False, filename=None, savepath=None,
               cbar_adj=True,
               fontsize=12,labelpad=10,
               **cbar_kwargs):
    """
    Plot an integrated intensity map over the given wavelength or wavenumber range
    :param data (LumiSpectrum): hyperspectral data
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param processed_data (np.ndarray)(optional): the registered data
    :return: intint (np.ndarray): the integrated intensity map
    """
    intint = get_intint(data, spectral_range=spectral_range, processed_data=processed_data)
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
    """
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(intint, data.axes_manager[0].scale, percent=frac_scalebar)

    # Plot the map
    fig,ax = plt.subplots()
    if cbar_adj:
        vmin, vmax = adjust_colorbar(intint,**cbar_kwargs)
        cmap = ax.imshow(intint, vmin=vmin, vmax=vmax, cmap='viridis')
    else:
        cmap = ax.imshow(intint, cmap='viridis')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                               borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                               fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(cmap, ax=ax, format=fmt)
    cbar.set_label('{} integrated intensity / a.u.'.format(data_type), fontsize=fontsize, labelpad=labelpad)
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
def int_map(original_data, wavelength, data_type,
            frac_scalebar=0.133335,
            processed_data=None,
            cbar_adj=True,
            fontsize=12,labelpad=10,
            savefig=False, figname=None, savefile=False, filename=None, savepath=None,
            **kwargs):
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
    """
    Spectr = original_data.axes_manager[2].axis
    if processed_data is not None:
        int_map = processed_data[:,:,abs(Spectr-wavelength).argmin()]
    else:
        int_map = original_data.data[:,:,abs(Spectr-wavelength).argmin()]
    """
    int_map = get_int(original_data, wavelength, processed_data=processed_data)
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(int_map, original_data.axes_manager[0].scale, percent=frac_scalebar)

    fig, ax = plt.subplots()
    if cbar_adj:
        vmin, vmax = adjust_colorbar(int_map, **kwargs)
        cmap = ax.imshow(int_map, vmin=vmin, vmax=vmax, cmap='viridis')
    else:
        cmap = ax.imshow(int_map, cmap='viridis')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                 borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                 fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    if data_type == 'PL':
        cbarlabel = 'PL intensity / a.u.'
    elif data_type == 'Raman':
        cbarlabel = 'Raman intensity / a.u.'
    elif data_type == 'PLRatio':
        cbarlabel = 'Normalized PL intensity'
    elif data_type == 'RamanRatio':
        cbarlabel = 'Normalized Raman intensity'
    cbar=fig.colorbar(cmap, ax=ax)
    cbar.set_label(cbarlabel, fontsize=fontsize, labelpad=labelpad)
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
#%% plot a ratio map of two intensity maps at the given wavelengths or wavenumbers
def ratio_map(original_data, wavelength1, wavelength2, data_type,
                frac_scalebar=0.133335,
                cbar_adj=True,
                fontsize=12,labelpad=10,
                savefig=False, figname=None, savefile=False, filename=None, savepath=None,
                **kwargs):
    """
    Plot a ratio map of two intensity maps at the given wavelengths or wavenumbers
    :param original_data:
    :param wavelength1:
    :param wavelength2:
    :param data_type:
    :param frac_scalebar:
    :param cbar_adj:
    :param fontsize:
    :param labelpad:
    :param savefig:
    :param figname:
    :param savefile:
    :param filename:
    :param savepath:
    :param kwargs:
    :return:
    """
    ratio = get_ratio(original_data, wavelength1, wavelength2)
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(ratio, original_data.axes_manager[0].scale, percent=frac_scalebar)
    fig, ax = plt.subplots()
    if cbar_adj:
        vmin, vmax = adjust_colorbar(ratio, **kwargs)
        cmap = ax.imshow(ratio, vmin=vmin, vmax=vmax, cmap='viridis')
    else:
        cmap = ax.imshow(ratio, cmap='viridis')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                    borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                    fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label(f'{data_type} intensity ratio / a.u.',fontsize=fontsize, labelpad=labelpad)
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
            np.savetxt(savepath+filename+'.txt', ratio)
    return ratio
#%% Plot colormap
def plot_colormap(data, scale=None, frac_scalebar=0.133335,
                  fontsize=12,labelpad=10,
                  cbar_adj=True,cbar_label=None, title=None, save_path=None,
                  sci_notation_cbar=False,  # whether to use scientific notation for the colorbar
                  **kwargs):
    """
    Plot the colormap of the hyperspectral data
    :param data (np.ndarray): the hyperspectral data
    :param scale (float)(optional): used for the scalebar of the colormap, default is None
    :param cbar_adj (bool)(optional): whether to adjust the colorbar, default is True
    :param cbar_label (str)(optional): the label of the colorbar, default is None
    :param save_path (str)(optional): the path to save the figure, default is None
    :return: None
    """
    fig, ax = plt.subplots()
    if cbar_adj:
        vmin, vmax = adjust_colorbar(data, **kwargs)
        cmap = ax.imshow(data, vmin=vmin, vmax=vmax)
    else:
        cmap = ax.imshow(data)
    if scale is not None:
        len_in_pix, length, width = get_scalebar_length(data, scale, percent=frac_scalebar)
        scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                   borderpad=0.1, sep=5, frameon=False, size_vertical=width,
                                   color='white', fontproperties={'size': 15, 'weight': 'bold'})
        ax.add_artist(scalebar)
        ax.set_axis_off()
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if sci_notation_cbar:
        cbar = fig.colorbar(cmap, ax=ax, format=fmt)
    else:
        cbar = fig.colorbar(cmap, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=fontsize, labelpad=labelpad)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=True, dpi=300)
    plt.show()
#%% Define gaussian fitting function for the triple Gaussian
from scipy.optimize import curve_fit
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
def triple_gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3):
    return (amp1 * np.exp(-(x - cen1)**2 / (2 * wid1**2)) +
            amp2 * np.exp(-(x - cen2)**2 / (2 * wid2**2)) +
            amp3 * np.exp(-(x - cen3)**2 / (2 * wid3**2)))
#%% jacobian conversion for the spectral data
def jacobian_conversion(data, wl,ylabel='PL intensity / a.u.',plot=False):
    """
    Convert the wavelength axis of the spectral data to energy axis, and also convert the intensity using the Jacobian conversion
    Note that h*c is ignored in the conversion, so the intensity is not in absolute units.
    :param data (LumiSpectrum): the spectral data
    :return: data3d (np.ndarray): the spectral data with the energy axis and converted intensity
             energy_axis (np.ndarray): the energy axis of the spectral data
    """
    # Convert the wavelength axis to energy axis
    h = 6.62607015e-34  # Planck constant in J.s
    c = 3.0e8  # Speed of light in m/s
    eV = 1.602176634e-19  # 1 eV in J
    energy_axis = h * c / (wl * 1e-9) / eV
    # Convert the intensity using the Jacobian conversion
    intensity = data / energy_axis**2
    spectrum_converted = [energy_axis, intensity]
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(energy_axis, intensity)
        plt.xlabel('Energy / eV')
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    return spectrum_converted
#%% Jacobian conversion for the hyperspectral data
def jacobian_conversion_hsdata(hsdata, copy=False, rewrite=False):
    """
    Convert the wavelength axis of the hyperspectral data to energy axis, and also convert the intensity using the Jacobian conversion
    Note that h*c is ignored in the conversion, so the intensity is not in absolute units.
    :param data (LumiSpectrum): the hyperspectral data
    :return: data3d (np.ndarray): the hyperspectral data with the energy axis and converted intensity
             energy_axis (np.ndarray): the energy axis of the hyperspectral data
    """
    wl = hsdata.axes_manager[2].axis
    # Convert the wavelength axis to energy axis
    h = 6.62607015e-34  # Planck constant in J.s
    c = 3.0e8  # Speed of light in m/s
    eV = 1.602176634e-19  # 1 eV in J
    energy_axis = h * c / (wl * 1e-9) / eV
    # Convert the intensity using the Jacobian conversion
    intensity = hsdata.data / energy_axis[None,None,:] **2
    if copy:
        hsdata_new = copy.deepcopy(hsdata)
        hsdata_new.data = intensity
        # renew the metadata
        hsdata_new.axes_manager[2].name = 'Energy'
        hsdata_new.axes_manager[2].units = 'eV'
        hsdata_new.axes_manager[2].axis = energy_axis
        hsdata_new.axes_manager[2].size = energy_axis.size
        return hsdata_new, energy_axis
    if rewrite:
        hsdata.data = intensity
        # renew the metadata
        hsdata.axes_manager[2].name = 'Energy'
        hsdata.axes_manager[2].units = 'eV'
        hsdata.axes_manager[2].axis = energy_axis
        hsdata.axes_manager[2].size = energy_axis.size
        return hsdata, energy_axis
    else:
        return intensity, energy_axis
#%% tripple gaussian fitting to extract parameters and errors
def gaussian_fit(data, jacobian=False, func_name='triple_gaussian',initial_guesses=None,bounds=(-np.inf,np.inf), **kwargs):
    """
    Fit the PL spectrum of a single pixel with the triple Gaussian peaks
    :param data (LumiSpectrum): the original (or reconstructed) hyperspectral data read by hyperspy
    :param processed_data (np.ndarray): the warped hyperspectral data
    :param initial_guesses (list): the initial guesses for the Gaussian parameters
    :return: params (np.ndarray): the Gaussian parameters,
                errors (np.ndarray): the errors of the Gaussian parameters
    """
    if jacobian:
        spectra, energy_axis = jacobian_conversion_hsdata(data, copy=False, rewrite=False)
    else:
        spectra = data.data
        energy_axis = data.axes_manager[2].axis
    if func_name == 'triple_gaussian':
        func = triple_gaussian
        fit_params = np.zeros((data.data.shape[0], data.data.shape[1], 9))
        fit_errors = np.zeros((data.data.shape[0], data.data.shape[1], 9))
    elif func_name == 'gaussian':
        func = gaussian
        fit_params = np.zeros((data.data.shape[0], data.data.shape[1], 3))
        fit_errors = np.zeros((data.data.shape[0], data.data.shape[1], 3))

    for i in range(data.data.shape[0]):
        for j in range(data.data.shape[1]):
            spectrum = spectra[i, j, :]
            try:
                # Fit the spectrum with the triple Gaussian function
                popt, cov = curve_fit(func, energy_axis, spectrum, p0=initial_guesses,bounds=bounds, **kwargs)
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
def plot_gaussian_fit(data, params, jacobian=False, func_name='triple_gaussian', px_YX=(0,0),
                      residual_plot=False,
                      fontsize=12,labelpad=10, labelsize=12,
                      save_path=None):
    """
    Plot the spectrum with the triple Gaussian fitting and the residuals at the given pixel
    :param original_data (LumiSpectrum): the original hyperspectral data
    :param params (np.ndarray): the Gaussian parameters
    :param px_YX (tuple): the coordinates of the pixel in the format (y,x)
    :param save_path (str)(optional): the path to save the figure
    :return: None
    """
    if jacobian:
        spectra, x_axis = jacobian_conversion_hsdata(data, copy=False, rewrite=False)
    else:
        spectra = data.data
        x_axis = data.axes_manager[2].axis
    spectrum = spectra[px_YX[0], px_YX[1], :]
    params_pixel = params[px_YX[0], px_YX[1], :]
    if func_name == 'triple_gaussian':
        fit1 = gaussian(x_axis, *params_pixel[:3])
        fit2 = gaussian(x_axis, *params_pixel[3:6])
        fit3 = gaussian(x_axis, *params_pixel[6:])
        fit = triple_gaussian(x_axis, *params_pixel)
    elif func_name == 'gaussian':
        fit = gaussian(x_axis, *params_pixel)
    residuals = spectrum - fit
    if not residual_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_axis, spectrum)
        if func_name == 'triple_gaussian':
            ax.plot(x_axis, fit, color='r')
            ax.plot(x_axis, fit1, color='y')
            ax.plot(x_axis, fit2, color='g')
            ax.plot(x_axis, fit3, color='c')
        elif func_name == 'gaussian':
            ax.plot(x_axis, fit, color='r')
        ax.set_xlabel('Energy / eV', fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel('PL intensity / a.u.', fontsize=fontsize, labelpad=labelpad)
        ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    if residual_plot:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].scatter(x_axis, spectrum)
        if func_name == 'triple_gaussian':
            ax[0].plot(x_axis, fit1, color='y')
            ax[0].plot(x_axis, fit2, color='g')
            ax[0].plot(x_axis, fit3, color='c')
            ax[0].plot(x_axis, fit, color='r')
        elif func_name == 'gaussian':
            ax[0].plot(x_axis, fit, color='r')
        ax[1].scatter(x_axis, residuals)
        ax[0].set_xlabel('Energy / eV', fontsize=fontsize, labelpad=labelpad)
        ax[0].set_ylabel('PL intensity / a.u.', fontsize=fontsize, labelpad=labelpad)
        ax[1].set_xlabel('Energy / eV', fontsize=fontsize, labelpad=labelpad)
        ax[1].set_ylabel('Data - Fit', fontsize=fontsize, labelpad=labelpad)
        ax[0].tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
        ax[1].tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
    return None
#%% Step2: Plot the integrated intensity map of the individual Gaussian peak over the entire spectral range
from scipy import integrate
def plot_intint_gaussian(original_data,params,cbar_label='PL integrated intensity / a.u.',
                         jacobian=False,
                         cbar_adj=True, ROI=None,
                         frac_scalebar=0.133335,
                         fontsize=12,labelpad=10,
                         sum_threshold=None,save_path=None,
                         **kwargs):
    """
    Plot the integrated intensity maps of the individual Gaussian peaks over the entire spectral range
    :param original_data (LumiSpectrum): the original hyperspectral data used to provide the scale information
    :param intint_gaussian (np.ndarray): the integrated intensity map of the individual Gaussian peaks
    :param cbar_label (str): the label of the colorbar
    :param ROI (tuple)(optional): the region of interest (ROI) in the format (x, y, width, height)
    :param save_path (str)(optional): the path to save the figure
    :return: None
    """
    # Initialize intensity maps for each Gaussian
    wls = original_data.axes_manager[2].axis
    map_shape = original_data.data.shape
    if jacobian:
        _, wls = jacobian_conversion_hsdata(original_data, copy=False, rewrite=False)
        # flip the wls axis if it is in descending order
        if wls[0] > wls[-1]:
            wls = wls[::-1]
    intint_gaussian = np.zeros(map_shape[:2])
    # Loop through each pixel in the spatial dimensions
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            #intint_gaussian[i, j] = gaussian_integral(params[i, j, 0], params[i, j, 2])
            intint_gaussian[i, j] = integrate.quad(gaussian, wls[0], wls[-1],
                                                   args=(params[i, j, 0], params[i, j, 1], params[i, j, 2]))[0]
            spectrum = original_data.data[i, j, :]
            if sum_threshold is not None:
                if np.sum(spectrum) < sum_threshold:
                    intint_gaussian[i, j] = 0 # use the sum of the intensity to filter out the bad pixels or background
    # integrate a single Gaussian peak
    if ROI is not None:
        intint_gaussian = intint_gaussian[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
    fig, ax = plt.subplots()
    if cbar_adj:
        vmin, vmax = adjust_colorbar(intint_gaussian, **kwargs)
        cmap = ax.imshow(intint_gaussian, vmin=vmin, vmax=vmax)
    else:
        cmap = ax.imshow(intint_gaussian)
    ax.set_axis_off()
    len_in_pix, length, width = get_scalebar_length(intint_gaussian, original_data.axes_manager[0].scale,
                                                     percent=frac_scalebar)
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                               borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                               fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(cmap, ax=ax, format=fmt)
    cbar.set_label(cbar_label, fontsize=fontsize, labelpad=labelpad)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

    return intint_gaussian
#%% Plot the centre of mass map
# Calculate and plot the COM of the PL spectrum for all pixels using vectorized operations
def plot_com_map(original_data, processed_data=None, spectral_range=None, params_ROI=None,data_type='PL',
                 cbar_adjust=True,
                 fontsize=12,labelpad=10,
                 frac_scalebar=0.133335,save_path=None,
                 **kwargs):
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
    com_map = get_com(original_data, spectral_range=spectral_range, processed_data=processed_data)
    '''
    if spectral_range is not None:
        lambda1, lambda2 = spectral_range
        # lambda1 and lambda2 are the wavelength range for the COM calculation
        index1 = abs(original_data.axes_manager[2].axis-lambda1).argmin()
        index2 = abs(original_data.axes_manager[2].axis-lambda2).argmin()
    else:
        index1 = 0
        index2 = original_data.data.shape[2] - 1
    if processed_data is not None:
        intensities = processed_data[:,:,index1:index2]
    else:
        intensities = original_data.data[:,:,index1:index2]
    wavelength = original_data.axes_manager[2].axis[index1:index2]
    # Calculating COM for all pixels using vectorized operations
    numerator = np.sum(wavelength * intensities, axis=2)
    denominator = np.sum(intensities, axis=2)
    com_map = numerator / denominator
    '''
    if data_type == 'PL':
        #convert wavelength to energy
        #com_map = 1239.8 / com_map
        cbarlabel = f'PL center of mass / {original_data.axes_manager[2].units}'
    elif data_type == 'Raman':
        cbarlabel = 'Raman center of mass / cm$^{-1}$'
    # check if the is NaN and replace it with zero
    com_map = np.nan_to_num(com_map)
    fig, ax = plt.subplots()
    if params_ROI is not None:
        com_map = com_map[params_ROI[1]:params_ROI[1]+params_ROI[3],params_ROI[0]:params_ROI[0]+params_ROI[2]]
    if cbar_adjust:
        vmin, vmax = adjust_colorbar(com_map,**kwargs)
        cmap = ax.imshow(com_map, vmin=vmin, vmax=vmax)
    else:
        cmap = ax.imshow(com_map)
    ax.set_axis_off()
    len_in_pix, length, width = get_scalebar_length(com_map, original_data.axes_manager[0].scale,
                                                     percent=frac_scalebar)
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                               borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                               fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    cbar = fig.colorbar(cmap, ax=ax, format='%.1f')
    cbar.set_label(cbarlabel, fontsize=fontsize, labelpad=labelpad)
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
#%% Interactive point selection on a 2D map
import sys
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication
def select_point(data):
    """
    Display a 2D array with matplotlib and let the user click one or more points.

    Parameters
    ----------
    data : np.ndarray
        2D array representing the map.
    cmap : str
        Colormap for imshow.

    Returns
    -------
    list of (int, int)
        List of (col, row) pixel coordinates in order of clicking.
        If only one point is clicked, a list with single [(col, row)] tuple is returned.
    """
    vmin, vmax = adjust_colorbar(data)

    # Ensure a Qt application exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create figure and canvas
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    cax = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title("Click points (Press Enter or right-click to finish)")

    # Show the figure
    canvas.show()

    # ginput waits for clicks on this figure
    pts = fig.ginput(0, timeout=0)

    # Close the figure
    canvas.close()
    fig.clear()

    if not pts:
        print("No points selected.")
        return None

    # Convert to integer pixel coordinates
    coords = [(int(round(x)), int(round(y))) for x, y in pts]
    print(f"Selected pixel: {coords}")
    return coords
#%% Mark points of interest on the map
def point_marker(map_data,original_data, XY,cbarlabel='Intensity / a.u.',frac_scalebar=0.133335,
                 fontsize=12,labelpad=10,colorseq='Dark2',colors_udef=None,
                 cbar_adjust=True,cbar_sci_notation=True,
                 save_path=None,
                 **kwargs):
    """
    Mark the points of interest on the map
    :param map_data (np.ndarray): the dataset of map
    :param original_data (LumiSpectrum): the original data read by hyperspy used to provide the scale information
    :param YX (list): the list of the coordinates of the points of interest (Y,X)
    :param cbarlabel (str): the label of the colorbar
    :param save_path (str)(optional): the path to save the figure
    """
    # define scalebar
    colors = mpl.color_sequences[colorseq]  # use the color sequence from matplotlib
    if colors_udef is not None:
        colors = colors_udef
    len_in_pix, length, width = get_scalebar_length(map_data, original_data.axes_manager[0].scale,
                                                     percent=frac_scalebar)
    fig,ax = plt.subplots()
    if cbar_adjust:
        vmin, vmax = adjust_colorbar(map_data, **kwargs)
        cmap = ax.imshow(map_data,vmin=vmin, vmax=vmax)
    else:
        cmap = ax.imshow(map_data)
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                               borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                               fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    #colors = ['m', 'k', 'b', 'r', 'c', 'g']
    for i in range(len(XY)):
        ax.plot(XY[i][0], XY[i][1], 'o', mfc='none', mec=colors[i],mew=3, markersize=15)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if cbar_sci_notation:
        cbar=fig.colorbar(cmap, ax=ax,format=fmt)
    else:
        cbar=fig.colorbar(cmap, ax=ax)
    cbar.set_label(cbarlabel, fontsize=fontsize, labelpad=labelpad)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

#%% Extract the spectrum at the points of interest and plot them on the same figure
def Spectrum_extracted(original_data, XY, data_type,
                       jacobian=False, xlabel_PL='Wavelength / nm',
                       major_locator=50,n_minor_locator=2,
                       fontsize=12,labelpad=10, labelsize=12,
                       sci_notation_y=False,
                       colorseq='Dark2', colors_udef=None,
                       x_lim=None,y_lim=None,save_path=None):
    """
    Plot the spectrum at the points of interest on the same figure
    :param original_data (LumiSpectrum): hyperspectral data read by Hyperspy
    :param XY (tuple or list): the coordinates of the points of interest (X, Y) or the list of the coordinates of the points of interest (X,Y)
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param x_lim (list)(optional): the x limit of the plot
    :param y_lim (list)(optional): the y limit of the plot
    :return: fig (matplotlib.figure.Figure): the figure object,
                ax (matplotlib.axes._axes.Axes): the axes object
    """
    colors = mpl.color_sequences[colorseq]
    Spectr = original_data.axes_manager[2].axis
    intensities = original_data.data
    if x_lim is not None:
        x1 = np.argmin(np.abs(Spectr - x_lim[0]))
        x2 = np.argmin(np.abs(Spectr - x_lim[1]))
        Spectr = Spectr[x1:x2]
        intensities = intensities[:,:,x1:x2]
    if colors_udef is not None:
        colors = colors_udef
    if not isinstance(XY, list):
        XY = [XY]
    #colors = ['orange', 'k', 'b', 'r', 'c', 'g'] # set the color map, here we use the RdPu colormap
    fig, ax = plt.subplots()
    for i in range(len(XY)):
        x = XY[i][0]
        y = XY[i][1]
        intensity = intensities[y, x, :]
        if jacobian:
            Spectr, intensity = jacobian_conversion(intensity, Spectr, plot=False)
        ax.plot(Spectr, intensity, color=colors[i])
    if data_type == 'PL':
        ax.set_xlabel(xlabel_PL,
                      fontsize=fontsize, labelpad=labelpad)
        plt.gca().xaxis.set_major_locator(MultipleLocator(major_locator))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
        plt.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    elif data_type == 'Raman':
        ax.set_xlabel('Raman shift / cm$^{-1}$', fontsize=fontsize, labelpad=labelpad)
        ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
        plt.gca().xaxis.set_major_locator(MultipleLocator(major_locator))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
        plt.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel('{} intensity / a.u.'.format(data_type), fontsize=fontsize, labelpad=labelpad)
    if sci_notation_y:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
#%% Plot an average spectrum
def avg_spectrum(data, data_type,
                 jacobian=False, xlabel_PL='Wavelength / nm',
                 major_locator=50,n_minor_locator=2,params_ROI=None,
                 sci_notation_y=True,
                 xlim=None, ylim=None,
                 fontsize=12,labelpad=10, labelsize=12,
                 savefig=False,figname=None,
                 savefile=False, filename=None, save_path=None):
    """
    Plot the average spectrum over the whole map
    :param data (LumiSpectrum): hyperspectral data
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param major_locator (int): the major locator for the x-axis
    :param n_minor_locator (int): the minor locator for the x-axis
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
    if xlim is not None:
        x1 = np.argmin(np.abs(Spectr - xlim[0]))
        x2 = np.argmin(np.abs(Spectr - xlim[1]))
        Spectr = Spectr[x1:x2]
        avg_intens = avg_intens[x1:x2]

    fig, ax = plt.subplots()
    if jacobian:
        Spectr, avg_intens = jacobian_conversion(avg_intens, Spectr, plot=False)
    ax.plot(Spectr, avg_intens)
    if data_type == 'PL':
        x_label = xlabel_PL
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    elif data_type == 'Raman':
        x_label = 'Raman shift / cm$^{-1}$'
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    ax.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.set_ylabel('{} intensity / a.u.'.format(data_type), fontsize=fontsize, labelpad=labelpad)
    ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    if sci_notation_y:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(save_path+figname+'.png', transparent=True, dpi=300)
    plt.show()
    if savefile:
        if filename is None:
            print('Please provide a file name')
        else:
            np.savetxt(save_path + filename + '.txt', avg_intens)
    return avg_intens

#%% Compare the spectra on the same figure, for example, the average spectrum before & after illumination
def plot_spectra(spc_list, wl_list, data_type,
                 jacobian=False,xlabel_PL='Wavelength / nm',
                 xlim=None,ylim=None,label_list=None,
                 text=False,text_coords=None,
                 major_locator=50,n_minor_locator=2,sci_notation_y=False,
                 fontsize=12,labelpad=10, labelsize=12,
                 colorstyle='default',
                 save_path=None):
    """
    Plot two spectra on the same figure
    :param spc_list (list): the list of the spectra
    :param wl_list (list): the list of the wavelengths of the spectra
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param save_path (str)(optional): the path to save the figure
    :param xlim (tuple)(optional): the x limit of the plot
    :param ylim (tuple)(optional): the y limit of the plot
    :param label_list (list)(optional): the list of the labels of the spectra
    :param major_locator (int): the major locator for the x-axis
    :param n_minor_locator (int): the number of minor locators for the x-axis
    :param save_path (str)(optional): the path to save the figure
    :return:
    """
    if xlim:
        for wl in wl_list:
            x1 = np.argmin(np.abs(wl - xlim[0]))
            x2 = np.argmin(np.abs(wl - xlim[1]))
            wl_list[wl_list.index(wl)] = wl[x1:x2]
            spc_list[wl_list.index(wl)] = spc_list[wl_list.index(wl)][x1:x2]

    fig, ax = plt.subplots()
    plt.style.use(colorstyle)  # set the color style
    for i in range(len(spc_list)):
        if jacobian:
            wl_list[i],spc_list[i] = jacobian_conversion(spc_list[i], wl_list[i], plot=False)
        ax.plot(wl_list[i], spc_list[i])
    if data_type == 'PL':
        x_label = xlabel_PL
    elif data_type == 'Raman':
        x_label = 'Raman shift / cm$^{-1}$'
    ax.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel('{} intensity / a.u.'.format(data_type), fontsize=fontsize, labelpad=labelpad)
    ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    ax.xaxis.set_major_locator(MultipleLocator(major_locator))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    if sci_notation_y:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if text:
        if label_list is None:
            print('Please provide the labels for the spectra')
        if text_coords is None:
            print('Please provide the coordinates for the text')
        for i in range(len(spc_list)):
            ax.text(text_coords[i][0], text_coords[i][1], label_list[i], transform=ax.transAxes,
                    fontsize=labelsize, va='bottom', ha='center',color=ax.lines[i].get_color())
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
#%% crop hyperspectral data by given ROI
def crop_hsdata(original_data, ROI):
    """
    Crop the hyperspectral data by given ROI
    :param original_data (LumiSpectrum): the original hyperspectral data read by Hyperspy
    :param ROI (tuple)(optional): the region of interest (ROI) in the format (x, y, width, height)
    :return: cropped_data (LumiSpectrum): the cropped hyperspectral data
    """
    x, y, width, height = ROI
    left_in_um = original_data.axes_manager[0].axis[x]
    top_in_um = original_data.axes_manager[1].axis[y]
    right_in_um = original_data.axes_manager[0].axis[x + width]
    bottom_in_um = original_data.axes_manager[1].axis[y + height]
    roi = hs.roi.RectangularROI(top=top_in_um, bottom=bottom_in_um, left=left_in_um, right=right_in_um)
    cropped_hsdata = roi(original_data)
    return cropped_hsdata
#%% Find local maxima in the spectrum based on scipy.signal.find_peaks
import scipy.signal as signal

def find_maxima(original_data, spectrum_data, data_type,
                major_locator=50,n_minor_locator=2, secondary_axis=False,
                xlim=None,
                fontsize=12,labelpad=10, labelsize=12,
                save_path=None,
                *args, **kwargs):
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
    spectral_axis = original_data.axes_manager[2].axis
    if xlim is not None:
        x1 = np.argmin(np.abs(spectral_axis - xlim[0]))
        x2 = np.argmin(np.abs(spectral_axis - xlim[1]))
        spectral_axis = spectral_axis[x1:x2]
        spectrum_data = spectrum_data[x1:x2]
    peaks, _ = signal.find_peaks(spectrum_data, *args, **kwargs)
    pps = np.array(spectral_axis[peaks])  # peak positions
    pint = np.array(spectrum_data[peaks])  # peak intensity
    plt.plot(spectral_axis, spectrum_data)
    plt.plot(pps, pint, 'x', label='Maxima')
    for p in range(len(pps)):
        plt.text(pps[p], pint[p]+2, np.round(pps[p]), fontsize=fontsize, rotation='vertical', va='bottom', ha='center')
    if data_type == 'PL':
        x_label = f'{original_data.axes_manager[2].name} / {original_data.axes_manager[2].units}'
        if secondary_axis:
            secx = plt.gca().secondary_xaxis('top', functions=(lambda2energy, energy2lambda))
            secx.set_xlabel('Energy / eV', fontsize=fontsize, labelpad=labelpad)
            secx.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
        plt.gca().xaxis.set_major_locator(MultipleLocator(major_locator))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    elif data_type == 'Raman':
        x_label = 'Raman shift / cm$^{-1}$'
        plt.gca().xaxis.set_major_locator(MultipleLocator(major_locator))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    plt.xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
    plt.ylabel('{} intensity / a.u.'.format(data_type), fontsize=fontsize, labelpad=labelpad)
    plt.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
    return pps

#%% Plot histogram(s) of the 2d map data
import scipy.stats as stats
def plot_hist(dataMap, labels=None, colorstyle='default',
              spread=True, bins=100, bins_range=None, x_label='Intensity / a.u.', save_path=None):
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
    fig, ax = plt.subplots()
    plt.style.use(colorstyle)  # set the color style
    if isinstance(dataMap,list):
        for i in range(len(dataMap)):
            mean, std = stats.norm.fit(dataMap[i].flatten())
            if bins_range is not None:
                Range = bins_range
            else:
                Range = (dataMap[i].flatten().min(), dataMap[i].flatten().max())
            if spread:
                # plot the corresponding normal distribution
                x = np.linspace(Range[0], Range[1], bins)
                y = stats.norm.pdf(x, mean, std)
                ax.plot(x, y)
                ax.hist(dataMap[i].flatten(), bins=bins, range=Range, color=ax.lines[i].get_color(), alpha=0.5, label=labels[i]+': mean={:.1f}, std={:.1f}'.format(mean, std))
            else:
                ax.hist(dataMap[i].flatten(), bins=bins, range=Range, color=colors[i], alpha=0.5, label=labels[i])
    else:
        mean, std = stats.norm.fit(dataMap.flatten())
        if bins_range is not None:
            Range = bins_range
        else:
            Range = (dataMap.flatten().min(), dataMap.flatten().max())
        if spread:
            # plot the corresponding normal distribution
            x = np.linspace(Range[0], Range[1], bins)
            y = stats.norm.pdf(x, mean, std)
            ax.plot(x, y)
            ax.hist(dataMap.flatten(), bins=bins, range=Range, color=ax.lines[0].get_color(), alpha=0.5, label=labels+': mean={:.1f}, std={:.1f}'.format(mean, std))
        else:
            ax.hist(dataMap.flatten(), bins=bins, range=Range, color=colors[0], alpha=0.5, label=labels)
    ax.set_xlabel(x_label, fontsize=12, labelpad=10)
    ax.set_ylabel('Frequency', fontsize=12, labelpad=10)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()
#%% Plot 2D histogram of the correlative data
def plot_2d_hist(data_list, bins=50,labels=['PL intensity / a.u.','Raman intensity / a.u.'],save_path=None):
    '''
    Plot 2D histogram of the intensity maps
    :param data_list (list): the list of 2 datasets, for example, the PL and Raman intensity maps
    :param bins (int): the number of bins, default=50
    :param labels (list): the labels of the x and y axes
    :param save_path (str)(optional): the path to save the figure
    '''
    data1_flat = data_list[0].flatten()
    data2_flat = data_list[1].flatten()
    histogram, x_edges, y_edges = np.histogram2d(data1_flat, data2_flat, bins=bins)
    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(x_edges, y_edges, histogram.T, cmap='viridis')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Probability density')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=True, dpi=300)
    plt.show()
#%% Plot the grayscale white light reflection image
def plot_img(img_data,ROI=None,adj_hist=False,frac_scalebar=0.133335,save_path=None):
    '''
    Plot the grayscale image
    :param img_data (DataFrame): the image data read from the txt file using pandas
    :param ROI (list): (optional) [y_low,y_high,x_low,x_high] (in microns) the region of interest
    :param adj_hist (bool): (optional) whether to adjust contrast of the image via histogram equalization
    :param save_path (str): (optional) the path to save the figure
    :return:
    '''
    pixel_size_x = (img_data.axes[1].astype(float).max()-img_data.axes[1].astype(float).min())/img_data.axes[1].size
    pixel_size_y = (img_data.axes[0].astype(float).max()-img_data.axes[0].astype(float).min())/img_data.axes[0].size
    len_in_pix, length, width = get_scalebar_length(img_data.values, pixel_size_x, percent=frac_scalebar)
    fig, ax = plt.subplots()
    if ROI is not None:
        y_low_idx = np.argmin(np.abs(img_data.axes[0].astype(float)-ROI[0]))
        y_high_idx = np.argmin(np.abs(img_data.axes[0].astype(float)-ROI[1]))
        x_low_idx = np.argmin(np.abs(img_data.axes[1].astype(float)-ROI[2]))
        x_high_idx = np.argmin(np.abs(img_data.axes[1].astype(float)-ROI[3]))
        img_output = np.flipud(img_data.values[y_high_idx:y_low_idx,x_low_idx:x_high_idx])
    else:
        img_output = np.flipud(img_data.values)
    if adj_hist is True:
        hist, bins = np.histogram(img_output.flatten(), bins=256, range=(0, 256))
        cum_counts = np.cumsum(hist)
        tot_counts = np.sum(hist)
        tot_counts_5 = tot_counts * 0.01
        tot_counts_95 = tot_counts * 0.99
        bin_edges = bins[np.where((cum_counts >= tot_counts_5) & (cum_counts <= tot_counts_95))]
        ax.imshow(img_output, cmap='gray',vmin=bin_edges[0],vmax=bin_edges[-1])
    else:
        ax.imshow(img_output, cmap='gray')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                               borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                               fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

''' 
#%% ===== Old code for image registration using ANTs ====
import ants
#%% Create a binary mask to Highlight the features
from skimage import morphology
def create_mask(map2d, value_threshold, min_size, area_threshold):
    """
    Create a binary mask based on the 2D map
    :param map2d (np.ndarray): the 2D map
    :param min_size (int): the minimum size of the object
    :param area_threshold (int): the area threshold
    :return: mask (np.ndarray): the binary mask
    """
    # convert the map with float values to grayscale image
    map2d_gray = np.rint((map2d/map2d.max())*255)
    # create a binary mask (for example, fiducial marker area is 0, the rest is 1)
    mask = morphology.remove_small_holes(morphology.remove_small_objects(map2d_gray > value_threshold, min_size=min_size),
                                         area_threshold=area_threshold)
    # convert the mask to uint8
    mask = mask.astype(np.uint8)
    return mask

#%% Image registration using ANTs with optional parameters
def ants_registration(fixed_img, moving_img,type_of_transform,random_seed=None,interpolator='nearestNeighbor',outprefix='C:/',
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
    ax.imshow(fixed_img, cmap='gray')
    ax.imshow(warped_moving, alpha=0.7, cmap='viridis')
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
def transform2hsi(original_fixed_data, original_moving_data, transform, interpolator='nearestNeighbor',
                  transform_type='fwdtransforms'):
    """
    Apply the transformation to the entire hyperspectral image dataset
    :param original_fixed_data (LumiSpectrum): the original hyperspectral image dataset
    :param original_moving_data (LumiSpectrum): the original hyperspectral image dataset
    :param transform (dict): the transformation matrix
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :return: warped_data (np.ndarray): the registered hyperspectral image dataset
    """
    warped_data = np.zeros(original_moving_data.data.shape)
    for i in range(original_fixed_data.data.shape[2]):
        fixed_data = ants.from_numpy(original_fixed_data.data[:,:,i])
        moving_data = ants.from_numpy(original_moving_data.data[:,:,i])
        warped_data[:,:,i] = ants.apply_transforms(fixed=fixed_data, moving=moving_data, transformlist=transform[transform_type],interpolator=interpolator).numpy()
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
'''