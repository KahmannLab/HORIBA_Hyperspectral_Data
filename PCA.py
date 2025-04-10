import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

#%% Get the explained variance ratio plot (scree plot)
def plot_evr(dataPCA, n_components):
    """
    Plot the explained variance ratio by PCA
    :param dataPCA (LumiSpectrum): hyperspectral data with PCA decomposition
    :param n_components: the number of principal components shown in the plot
    :return: None
    """
    evr = dataPCA.get_explained_variance_ratio()
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
def PC_spc(dataPCA,component_index, data_type,save_path=None):
    """
    Plot the spectrum of the individual principal component
    :param dataPCA (LumiSpectrum): hyperspectral data with PCA decomposition
    :param component_index (int): the index of the principal component starting from 0 (Note the index given by Hyperspy is starting from 0, not 1)
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :param save_path (str)(optional): the path to save the figure
    """
    PCs = dataPCA.get_decomposition_factors()
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
def rec_data(dataPCA, num_PCs, single_PC=None):
    """
    Reconstruct the data using the given number of PCs or a certain PC
    :param dataPCA (LumiSpectrum): hyperspectral data with PCA decomposition
    :param num_PCs (int)(optional): the number of principal components or the index of the certain component starting from 0
    :param single_PC (tuple)(optional): if True, the single component is used.
    :return: rec_data (np.ndarray): the reconstructed data
    """
    if single_PC is True:
        rec_data = dataPCA.get_decomposition_model(components=[num_PCs])
    else:
        rec_data = dataPCA.get_decomposition_model(components=num_PCs)
    return rec_data