import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#%% New: PCA from sklearn for hypserpy data
# convert hyperspectral data to a 2D array where each row is a spectrum
def stack_spectra_columnwise(cube):
    """
    Stack all spectra vertically into a 2D array.
    Input: cube of shape (ny, nx, nspec)
    Output: array of shape (nspec, ny*nx), column-by-column stacking
    """
    ny, nx, nspec = cube.shape
    return cube.transpose(1, 0, 2).reshape(nx * ny, nspec)
# convert the stacked spectra back to a 3D cube
def unstack_spectra_columnwise(flat, ny, nx):
    """
    Restore stacked spectra back to a 3D cube (ny, nx, nspec)
    Input: stacked array of shape (nspec, ny*nx)
    """
    nspec = flat.shape[1]
    return flat.reshape(nx, ny, nspec).transpose(1, 0, 2)

#%%
from sklearn.decomposition import PCA

def sklearn_PCA(data,ScreePlot=False,n_PCs=None,*args, **kwargs):
    # flatten the datacube read by hyperspy
    flat_data = stack_spectra_columnwise(data.data)
    # apply PCA
    pca = PCA(*args, **kwargs)
    pca.fit(flat_data)
    # save all components in an array
    component_spectra = pca.components_
    if ScreePlot:
        # plot the explained variance ratio vs the number of components
        EVR = pca.explained_variance_ratio_
        N_components = np.arange(1, len(EVR) + 1)
        if n_PCs is not None:
            N_components = N_components[:n_PCs]
            EVR = EVR[:n_PCs]
        plt.plot(N_components, EVR, 'ro', markersize=5)
        ''''
        # using majorlocator and minor locator to set the x-axis ticks
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))
        '''
        plt.xticks(N_components, fontsize=10)
        # plt.yticks(fontsize=10)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.yscale('log')
        plt.title('Explained variance ratio by PCA')
        plt.tight_layout()
        plt.show()

    return pca, component_spectra
#%% plot the PCA component spectra
def plot_PCs(component_spectra, component_idx=6,
             x_label='Raman shift / cm$^{-1}$', y_label='Intensity / a.u.',
             savefig=False, figname=None, savepath=None):
    """
    Plot the PCA component spectra.

    Parameters:
        component_spectra : array-like, shape (n_components, n_features)
            The PCA component spectra.
        component_idx : int or list of int, optional
            If int, the first n components will be plotted.
            If list, specific components will be plotted.

    Returns:
        None
    """
    if isinstance(component_idx, int):
        # Plot the first n components
        indices = list(range(component_idx))
    elif isinstance(component_idx, list):
        # Plot specific components
        indices = component_idx
    else:
        raise ValueError("component_idx must be an int or a list of ints.")
    for idx in indices:
        fig, ax = plt.subplots()
        ax.plot(component_spectra[idx], label=f'PC {idx + 1}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        plt.tight_layout()
        if savefig:
            if figname is None:
                print(f"Warning: No figure name provided, using default 'PC{idx+1}'.")
                figname = f'PC{idx+1}'
            else:
                figname = figname+f'_PC{idx+1}'
            if savepath is not None:
                plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
            else:
                plt.savefig(figname+'.png', transparent=True, dpi=300)
                print("Warning: No save path provided, saving in the current directory.")

        plt.show()
#%% reconstruct the data
def reconstruct_data(data, pca, component_idx=None, component_list=None):
    """
       Reconstruct data from PCA scores using a subset of components.

       Parameters:
           pca : trained sklearn PCA object
           X_pca : array-like, shape (n_samples, n_components)
               PCA-transformed data (scores)
           n_first : int, optional
               Number of first components to use (indices 0 to n_first-1)
           component_list : list of int, optional
               Specific component indices to use

       Returns:
           X_reconstructed : array-like, shape (n_samples, n_features)
               Data reconstructed using selected components
       """
    flat_data = stack_spectra_columnwise(data.data)
    data_pca = pca.transform(flat_data)
    if component_idx is not None:
        # Select first n_first components
        selected_indices = list(range(component_idx))
    elif component_list is not None:
        # Select specific components
        selected_indices = component_list
    else:
        raise ValueError("Either the index of component or component_list must be provided.")

    # Extract scores and components
    data_pca_sel = data_pca[:, selected_indices]
    components_sel = pca.components_[selected_indices]

    # Reconstruct data
    data_reconstructed = np.dot(data_pca_sel, components_sel) + pca.mean_
    datacube_reconstructed = unstack_spectra_columnwise(data_reconstructed, data.data.shape[0], data.data.shape[1])
    return datacube_reconstructed

#%% Score map
# get the ideal scalebar length
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
def score_map(data, pca, component_index,*args, **kwargs):
    """
    Plot the score map of the given PC
    :param data (LumiSpectrum): hyperspectral data
    :param pca: trained sklearn PCA object
    :param component_index (int): the index of the principal component starting from 1
    """
    flat_data = stack_spectra_columnwise(data.data)
    data_pca = pca.transform(flat_data)
    # reshape the PCA scores to match the original data shape
    data_pca_reshaped = unstack_spectra_columnwise(data_pca, data.data.shape[0], data.data.shape[1])
    score_map = data_pca_reshaped[:,:, component_index - 1]  # component_index is starting from 1, so we need to subtract 1 for zero-based indexing
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(score_map, data.axes_manager[0].scale, *args, **kwargs)

    fig, ax = plt.subplots()
    cmap = ax.imshow(score_map, cmap='viridis')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Score')
    ax.set_title('Score map of the {}. PC'.format(component_index))  # index shown in the title is starting from 1
    plt.tight_layout()
    plt.show()
'''    
#%% Old PCA functions based on Hyperspy PCA
# Get the explained variance ratio plot (scree plot)
def plot_evr(dataPCA, n_components,save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

#%% Plot the PC spectra
def PC_spc(dataPCA,component_index, data_type,major_locator=50,n_minor_locator=2,save_path=None):
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
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    elif data_type == 'Raman':
        xlabel = 'Raman shift (cm$^{-1}$)'
        ylabel = 'Raman intensity (counts)'
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_title('{} spectrum based on the {}. PC'.format(data_type,component_index+1)) # the index shown in the title is starting from 1
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

#%% Plot the loading map of the given PC
def loading_map(dataPCA, component_index,ROI=None,save_path=None):
    """
    Plot the loading map of the given PC
    :param dataPCA: the data after PCA decomposition
    :param component_index: the index of the principal component starting from 1
    :param ROI(list): the region of interest to be shown in the loading map[y,y+dy,x,x+dx]
    :return:
    """
    loading_maps = dataPCA.get_decomposition_loadings()
    loading_map = loading_maps.data[component_index-1,:,:]
    if ROI is not None:
        loading_map = loading_map[ROI[0]:ROI[1], ROI[2]:ROI[3]]
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
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(cmap, ax=ax,format=fmt)
    cbar.set_label('Loading value')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,transparent=True,dpi=300)
    plt.show()

    return loading_map
#%% Reconstruct the data using given number of PCs or a certain PC
def rec_data(dataPCA, num_PCs, single_PC=False):
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

#%% Reconstruction animation
from matplotlib.animation import FuncAnimation
def rec_spc_animation(dataPCA, px, start_nPCs, end_nPCs=1, save_path='C:/animation.gif', data_type='PL',
                      major_locator=50, n_minor_locator=2,
                      writer='pillow',fps=10):
    """
    Create an animation of the data reconstruction using different number of PCs
    :param dataPCA:
    :param px: the pixel to be reconstructed (x,y)
    :param start_nPCs:
    :param end_nPCs:
    :param data_type:
    :param save_path:
    :param writer:
    :param fps:
    :return:
    """
    wl = dataPCA.axes_manager[2].axis
    rec_spcs = []
    for i in range(start_nPCs, end_nPCs+1, -1):
        rec_data = dataPCA.get_decomposition_model(components=i)
        rec_spc = rec_data.data[px[0], px[1], :]
        rec_spcs.append(rec_spc)
    rec_spcs = np.array(rec_spcs)
    fig, ax = plt.subplots(figsize=(8, 6))

    def update_plot(frame):
        ax.clear()
        spc = rec_spcs[frame]
        ax.plot(wl, spc)
        if data_type == 'PL':
            ax.set_xlabel('Wavelength (nm)', fontsize=12, labelpad=10)
            ax.set_ylabel('PL intensity (counts)', fontsize=12, labelpad=10)
        elif data_type == 'Raman':
            ax.set_xlabel('Raman shift (cm$^{-1}$)', fontsize=12, labelpad=10)
            ax.set_ylabel('Raman intensity (counts)', fontsize=12, labelpad=10)
        ax.set_title('Reconstructed spectrum using the first {} PC(s)'.format(start_nPCs - frame),fontsize=12, pad=10)
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_locator))

    ani = FuncAnimation(fig, update_plot, frames=len(rec_spcs), repeat=False)
    ani.save(save_path, writer=writer, fps=fps)
'''