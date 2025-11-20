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

def sklearn_PCA(data,ScreePlot=False,n_PCs=None,
                saveplot=False, figname=None, savepath=None,
                *args, **kwargs):
    """
    Perform PCA using sklearn on hyperspectral data.
    :param data (np.ndarray): hyperspectral data
    :param ScreePlot (bool): whether to plot the explained variance ratio
    :param n_PCs (int, optional): number of principal components to consider for the scree plot
    :param saveplot (bool): whether to save the scree plot
    :param figname (str): name of the figure to save
    :param savepath (str): path to save the figure
    :param args : additional arguments for sklearn PCA
    :param kwargs: additional arguments for sklearn PCA
    :return:
    pca : trained sklearn PCA object
    component_spectra : array-like, shape (n_components, n_features)
    """
    # flatten the datacube to 2D (pixels × spectrum)
    flat_data = stack_spectra_columnwise(data)
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
        # using majorlocator and minor locator to set the x-axis ticks
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))
        plt.tick_params(which='both', direction='in', right=True, top=True)
        # plt.yticks(fontsize=10)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.yscale('log')
        plt.title('Explained variance ratio by PCA')
        plt.tight_layout()
        if saveplot:
            if figname is None:
                print("Warning: No figure name provided, using default 'PCA_scree_plot'.")
                savename = 'PCA_scree_plot'
            else:
                savename = figname
            if savepath is not None:
                plt.savefig(savepath+savename+'.png', transparent=True, dpi=300)
            else:
                plt.savefig(savename+'.png', transparent=True, dpi=300)
                print("Warning: No save path provided, saving in the current directory.")
        plt.show()

    return pca, component_spectra
#%% plot the PCA component spectra
def plot_PCs(component_spectra, x_axis, component_idx,
             x_label='Raman shift / cm$^{-1}$', y_label='Intensity / a.u.',
             fontsize=12,labelpad=10, labelsize=12,
             savefig=False, figname=None, savepath=None):
    """
    Plot the PCA component spectra.

    Parameters:
        component_spectra : array-like, shape (n_components, n_features)
            The PCA component spectra.
        x_axis : array-like, shape (n_features,)
            The x-axis values.
        component_idx : int or list of int, optional
            If int, the first n components will be plotted.
            If list, specific components will be plotted, counting from 1.
        x_label : str, optional
            The label of the x-axis.
        y_label : str, optional
            The label of the y-axis.
        fontsize : int, optional
            The font size of the labels.
        labelpad : int, optional
            The labelpad of the labels.
        labelsize : int, optional
            The label size of the ticks.
        savefig : bool, optional
            Whether to save the figure.
        figname : str, optional
            The name of the figure to save.
        savepath : str, optional
            The path to save the figure.
    Returns:
        None
    """
    if isinstance(component_idx, int):
        # Plot the first n components
        indices = list(range(component_idx))
    elif isinstance(component_idx, list):
        # Plot specific components
        indices = [idx - 1 for idx in component_idx]  # Convert to zero-based index
    else:
        raise ValueError("component_idx must be an int or a list of ints.")
    for idx in indices:
        fig, ax = plt.subplots()
        ax.plot(x_axis,component_spectra[idx], label=f'PC {idx+1}')
        ax.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(y_label, fontsize=fontsize, labelpad=labelpad)
        # set the tick fontsize
        plt.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
        ax.legend()
        plt.tight_layout()
        if savefig:
            if figname is None:
                print(f"Warning: No figure name provided, using default 'PC{idx}'.")
                savename = f'PC{idx+1}'
            else:
                savename = figname+f'_PC{idx+1}'
            if savepath is not None:
                plt.savefig(savepath+savename+'.png', transparent=True, dpi=300)
            else:
                plt.savefig(savename+'.png', transparent=True, dpi=300)
                print("Warning: No save path provided, saving in the current directory.")

        plt.show()
#%% reconstruct the data
def reconstruct_data(data, pca, component_idx=None, component_list=None):
    """
       Reconstruct data from PCA scores using a subset of components.

       Parameters:
           data : array-like, shape (n_samples, n_features)
           pca : trained sklearn PCA object
           component_idx : int, optional
                   Number of components to use
           component_list : list of int, optional
               Specific component indices to use counting from 1

       Returns:
              datacube_reconstructed : np.ndarray
       """
    flat_data = stack_spectra_columnwise(data)
    data_pca = pca.transform(flat_data)
    if component_idx is not None:
        # Select first n_first components
        selected_indices = list(range(component_idx))
    elif component_list is not None:
        # Select specific components
        selected_indices = [comp - 1 for comp in component_list]  # Convert to zero-based index
    else:
        raise ValueError("Either the index of component or component_list must be provided.")

    # Extract scores and components
    data_pca_sel = data_pca[:, selected_indices]
    components_sel = pca.components_[selected_indices]

    # Reconstruct data
    data_reconstructed = np.dot(data_pca_sel, components_sel) + pca.mean_
    datacube_reconstructed = unstack_spectra_columnwise(data_reconstructed, data.shape[0], data.shape[1])

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
def score_map(data, pca, component_index,
              px_size,
              cbar_adjust=False,
              fontsize=12,labelpad=10,
              *args, **kwargs):
    """
    Plot the score map of the given PC
    :param data (np.ndarray): hyperspectral data
    :param pca: trained sklearn PCA object
    :param component_index (int): the index of the principal component starting from 1
    :param px_size (float): pixel size in micrometers
    :param cbar_adjust (bool): whether to adjust the colorbar limits based on the data histogram, default is False
    :param fontsize (int): font size for the scalebar label, default is 12
    :param labelpad (int): labelpad for the scalebar label, default is 10
    :param args: additional positional arguments for get_scalebar_length function
    :param kwargs: additional keyword arguments for get_scalebar_length function
    :return: score_map (2D np.ndarray): the score map of the given PC
    """
    flat_data = stack_spectra_columnwise(data)
    data_pca = pca.transform(flat_data)
    # reshape the PCA scores to match the original data shape
    data_pca_reshaped = unstack_spectra_columnwise(data_pca, data.shape[0], data.shape[1])
    score_map = data_pca_reshaped[:,:, component_index - 1]  # component_index is starting from 1, so we need to subtract 1 for zero-based indexing
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(score_map, px_size, *args, **kwargs)

    fig, ax = plt.subplots()
    if cbar_adjust:
        vmin, vmax = adjust_colorbar(score_map,**kwargs)
        cmap = ax.imshow(score_map, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        cmap = ax.imshow(score_map, cmap='viridis')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                                borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                                fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(cmap, ax=ax, format=fmt)
    cbar.set_label('Score', fontsize=fontsize, labelpad=labelpad)
    #ax.set_title('Score map of the {}. PC'.format(component_index))  # index shown in the title is starting from 1
    plt.tight_layout()
    plt.show()

    return score_map
#%% Option 2: applying PCA via scipy
from scipy.linalg import svd
def svd_PCA(data, ScreePlot=False, n_PCs=None, full_matrices=False):
    """
    Perform PCA using SVD.
    :param data (np.ndarray): hyperspectral data
    :param ScreePlot (bool): whether to plot the explained variance ratio
    :param n_PCs (int, optional): number of principal components to consider for the scree plot
    :param full_matrices (bool): whether to compute full matrices U and Vt
    :return: U, S, Vt, component_spectra, explained_variance_ratio
    """
    # flatten hyperspy datacube to 2D (pixels × spectrum)
    flat_data = stack_spectra_columnwise(data)
    # mean center
    X = flat_data - flat_data.mean(axis=0)

    # perform SVD: X = U Σ Vt
    U, S, Vt = svd(X, full_matrices=full_matrices)

    # Vt rows are principal components (like sklearn's pca.components_)
    component_spectra = Vt

    # explained variance (match sklearn's pca.explained_variance_ratio_)
    n_samples = X.shape[0]
    explained_variance = (S**2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var

    if ScreePlot:
        EVR = explained_variance_ratio
        N_components = np.arange(1, len(EVR) + 1)
        if n_PCs is not None:
            N_components = N_components[:n_PCs]
            EVR = EVR[:n_PCs]

        import matplotlib.pyplot as plt
        plt.plot(N_components, EVR, 'ro', markersize=5)
        # set major and minor locators for x-axis
        plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(1))
        plt.tick_params(which='both', direction='in', right=True, top=True)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.yscale('log')
        plt.title('Explained variance ratio by SVD')
        plt.tight_layout()
        plt.show()

    return U, S, Vt, component_spectra, explained_variance_ratio
#%% reconstruct the data using SVD
def reconstruct_data_svd(data, U, S, Vt, component_idx=None, component_list=None):
    """
    Reconstruct data from SVD scores using a subset of components.
    :param data (np.ndarray): hyperspectral data
    :param U (np.ndarray): left singular vectors
    :param S (np.ndarray): singular values
    :param Vt (np.ndarray): right singular vectors
    :param component_idx : int, optional, the number of components to use, default: None
    :param component_list : list of int, optional, the specific component indices to use counting from 1, default: None
    :return:
    datacube_reconstructed : np.ndarray
    """
    flat_data = stack_spectra_columnwise(data)
    X = flat_data - flat_data.mean(axis=0)

    if component_idx is not None:
        selected_indices = list(range(component_idx))
    elif component_list is not None:
        selected_indices = [comp - 1 for comp in component_list]  # Convert to zero-based index
    else:
        raise ValueError("Either component_idx or component_list must be provided.")

    # Use selected components: X ≈ U_k Σ_k Vt_k
    U_sel = U[:, selected_indices]
    S_sel = np.diag(S[selected_indices])
    Vt_sel = Vt[selected_indices, :]

    X_reconstructed = U_sel @ S_sel @ Vt_sel + flat_data.mean(axis=0)

    datacube_reconstructed = unstack_spectra_columnwise(
        X_reconstructed, data.shape[0], data.shape[1]
    )
    return datacube_reconstructed

#%% score map
def score_map_svd(data, U, S, component_index,
                  px_size,
                  cbar_adjust=False, *args, **kwargs):
    """
    Plot the score map using SVD results.
    :param data (np.ndarray): hyperspectral data
    :param U (np.ndarray): left singular vectors
    :param S (np.ndarray): singular values
    :param component_index (int): the index of the principal component starting from 1
    :param px_size (float): pixel size in micrometers
    :param cbar_adjust (bool): whether to adjust the colorbar limits based on the data histogram, default is False
    :param args: additional positional arguments for get_scalebar_length function
    :param kwargs: additional keyword arguments for get_scalebar_length function
    :return: score_map (2D np.ndarray): the score map of the given PC
    """
    flat_data = stack_spectra_columnwise(data)
    X = flat_data - flat_data.mean(axis=0)

    # Project onto PC
    scores = U[:, component_index - 1] * S[component_index - 1]

    # reshape back
    scores_reshaped = unstack_spectra_columnwise(
        scores[:, np.newaxis], data.shape[0], data.shape[1]
    )[:, :, 0]

    # scale bar
    len_in_pix, length, width = get_scalebar_length(
        scores_reshaped, px_size, *args, **kwargs
    )

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fig, ax = plt.subplots()
    if cbar_adjust:
        vmin, vmax = adjust_colorbar(scores_reshaped,**kwargs)
        cmap = ax.imshow(scores_reshaped, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        cmap = ax.imshow(scores_reshaped, cmap='viridis')
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(
        ax.transData, len_in_pix, str(length) + ' μm', 4,
        pad=1, borderpad=0.1, sep=5, frameon=False,
        size_vertical=width, color='white',
        fontproperties={'size': 15, 'weight': 'bold'}
    )
    ax.add_artist(scalebar)
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Score')
    ax.set_title(f'Score map of the {component_index}. PC')
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