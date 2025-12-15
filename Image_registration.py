import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ants
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import functools
import copy
#%% load images in txt format
def load_image_txt(file_path):
    image = np.loadtxt(file_path)
    return image
#%% mask to highlight the features of the image
@functools.wraps(ants.get_mask)
def create_mask(image, low_thresh=None, high_thresh=None, cleanup=2,plot=True, **kwargs):
    ants_img = ants.from_numpy(np.transpose(image))
    mask = ants.get_mask(ants_img, low_thresh=low_thresh, high_thresh=high_thresh, cleanup=cleanup, **kwargs)
    if plot:
        ants.plot(mask)
    return mask

def plot_img(image):
    plt.imshow(image, cmap='gray')
    plt.xlabel('Position / pixel')
    plt.ylabel('Position / pixel')
    plt.tight_layout()
    plt.show()
#%%
def return_overlap(moving_img, threshold=0, plot=False):
    """
    Compute and optionally visualize the overlapping region between two aligned images.
    The overlap is defined as pixels in the moving images with intensity above a given threshold, default is 0.

    Parameters
    ----------
    moving_img : np.ndarray
        The moving image after registration.
    threshold : float
        Intensity threshold to define valid pixels (default=0).
    plot : bool
        If True, visualize the overlap mask.

    Returns
    -------
    overlap_mask : np.ndarray (bool)
        Binary mask of overlapping valid region.
    bbox : tuple or None
        (ymin, ymax, xmin, xmax) bounding box of overlap, or None if no overlap.
    """
    mask1 = moving_img > threshold
    overlap_mask = mask1

    coords = np.argwhere(overlap_mask)
    if coords.size == 0:
        bbox = None
    else:
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0) + 1
        bbox = (ymin, ymax, xmin, xmax)

    if plot:
        plt.imshow(overlap_mask, cmap='gray')
        plt.title("Overlap Mask")
        plt.axis('off')
        plt.show()

    return overlap_mask, bbox

def crop_to_overlap(data, bbox):
    """
    Crop a 2D or 3D array to a given bounding box (from return_overlap()).

    Parameters
    ----------
    data : np.ndarray
        The data to crop (2D image or 3D hyperspectral cube).
    bbox : tuple
        Bounding box (ymin, ymax, xmin, xmax).

    Returns
    -------
    cropped_data : np.ndarray
        Cropped data.
    """
    if bbox is None:
        raise ValueError("No overlap bounding box provided for cropping.")
    ymin, ymax, xmin, xmax = bbox
    return data[ymin:ymax, xmin:xmax, :]

#%% register images using ANTsPy
# Image registration using ANTs with optional parameters: https://antspy.readthedocs.io/en/latest/registration.html
@functools.wraps(ants.registration)
@functools.wraps(ants.apply_transforms)
def img_registration(fixed_img, moving_img,fixed_mask=None,moving_mask=None,type_of_transform='antsRegistrationSyNRepro[r]',
                     random_seed=None,interpolator='nearestNeighbor',
                      savefig=False,figname=None,savepath=None,
                     return_overlap_flag=False, overlap_threshold=0,
                     plot_overlap=False,
                     **kwargs):
    """
    Image registration using ANTs
    :param fixed_img (np.ndarray): the fixed image
    :param moving_img (np.ndarray): the moving image
    :param type_of_transform (str): the type of the transformation supported by ANTs, default is 'antsRegistrationSyN[r]' (rigid)
    :param random_seed (int)(optional): the random seed, used to improve the reproducibility of the registration
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :param fixed_mask (np.ndarray)(optional): the mask of the fixed image, if None, the entire image will be used
    :param moving_mask (np.ndarray)(optional): the mask of the moving image, if None, the entire image will be used
    :param savefig (bool)(optional): whether to save the figure, default is False
    :param figname (str)(optional): the name of the figure to be saved, if None, the figure will not be saved
    :param savepath (str)(optional): the path to save the figure, if None, the figure will not be saved
    :param return_overlap : bool
        If True, compute and return overlap mask and bounding box.
    :param overlap_threshold : float
        Intensity threshold for overlap (default 0).
    :return: transform (dict): the transformation matrix
        warped_moving (np.ndarray): the registered moving image
    """
    ants_fixed_img = ants.from_numpy(fixed_img)
    ants_moving_img = ants.from_numpy(moving_img)
    '''
    if fixed_mask is not None:
        fixed = fixed_mask
    else:
        fixed = ants.from_numpy(fixed_img)
    if moving_mask is not None:
        moving = moving_mask
    else:
        moving = ants.from_numpy(moving_img)
    '''
    transform = ants.registration(fixed=ants_fixed_img, moving=ants_moving_img, mask=fixed_mask,
                                  moving_mask=moving_mask, type_of_transform=type_of_transform,random_seed=random_seed,**kwargs)
    warped_moving = ants.apply_transforms(fixed=ants_fixed_img, moving=ants_moving_img, transformlist=transform['fwdtransforms'],
                                          interpolator=interpolator,**kwargs).numpy()
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

    if return_overlap_flag:
        overlap_mask, bbox = return_overlap(
            warped_moving, threshold=overlap_threshold, plot=plot_overlap
        )
        return transform, warped_moving, overlap_mask, bbox

    return transform, warped_moving

#%% register the dataset
def transform2dataset(fixed_data, moving_data, transform,
                      transform_type = 'fwdtransforms',
                      interpolator='nearestNeighbor',
                      crop_to_overlap_flag=False, bbox=None, overlap_threshold=0,
                      **kwargs):
    """
    Apply the transformation to a 3d dataset
    :param fixed_data (np.ndarray): the fixed image
    :param moving_data (np.ndarray): the moving image
    :param transform (dict): the transformation matrix
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :param transform_type (str): the type of the transformation, default is 'fwdtransforms'
    :param crop_to_overlap_flag (bool): whether to crop the registered data to the overlapping region, default is False
    :param bbox (tuple): the bounding box of the overlapping region, default is None
    :param overlap_threshold (float): the intensity threshold for overlap, default is 0
    :return: warped_data (np.ndarray): the registered image
    """
    warped_data = np.zeros(moving_data.shape)
    for i in range(moving_data.shape[2]):
        fixed = ants.from_numpy(fixed_data[:,:,i])
        moving = ants.from_numpy(moving_data[:,:,i])
        warped_data[:,:,i] = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transform[transform_type],
                                                   interpolator=interpolator, **kwargs).numpy()
    # Crop if requested
    if crop_to_overlap_flag:
        if bbox is None:
            print("⚠️ No bounding box provided for cropping. Computing overlap from data.")
            _, bbox = return_overlap(warped_data.sum(axis=2), threshold=overlap_threshold, plot=False)

        warped_data = crop_to_overlap(warped_data, bbox)
    return warped_data
#%% register individual map
def transform2map(fixed_map, moving_map, transform,
                  interpolator='nearestNeighbor', transform_type='fwdtransforms',
                  crop_to_overlap_flag=False, bbox=None, overlap_threshold=0,
                  **kwargs):
    """
    Apply the transformation to a 2d map/image
    :param fixed_data (np.ndarray): the fixed image
    :param moving_data (np.ndarray): the moving image
    :param transform (dict): the transformation matrix generated by antsRegistration
    :param interpolator (str)(optional): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :param transform_type (str): the type of the transformation, default is 'fwdtransforms'
    :param crop_to_overlap_flag (bool): whether to crop the registered data to the overlapping region, default is False
    :param bbox (tuple): the bounding box of the overlapping region, default is None
    :param overlap_threshold (float): the intensity threshold for overlap, default is 0
    :return: warped_data (np.ndarray): the registered image
    """
    fixed = ants.from_numpy(fixed_map)
    moving = ants.from_numpy(moving_map)
    warped_map = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transform[transform_type],
                                       interpolator=interpolator, **kwargs).numpy()
    # Crop if requested
    if crop_to_overlap_flag:
        if bbox is None:
            print("⚠️ No bounding box provided for cropping. Computing overlap from data.")
            _, bbox = return_overlap(warped_map, threshold=overlap_threshold, plot=False)
        warped_map = crop_to_overlap(warped_map, bbox)
    return warped_map

#%% Upsample the image to fit the similar pixel size
import cv2
def resample_img(img, pixel_size, reference_pixel_size,plot=True):
    '''
    :param img (ndarray):
    :param pixel_size (float): pixel size of the input image, should be in the same unit as reference_pixel_size
    :param reference_pixel_size (float): pixel size after resampling, should be in the same unit as pixel_size
    :param plot:
    :return: resampled (ndarray): the resampled image
    '''
    # Compute scale factor (how much bigger the pixels should be)
    scale = pixel_size / reference_pixel_size
    if scale < 1: # Downsample
        interp = cv2.INTER_AREA
        title = "Downsampled image"
    elif scale > 1: # Upsample
        interp = cv2.INTER_LINEAR
        title = "Upsampled image"

    # New image size (width, height)
    new_size = (
        int(round(img.shape[1] * scale)),  # width
        int(round(img.shape[0] * scale))  # height
    )

    # Upsample or downsample the image using interpolation
    resampled = cv2.resize(img, new_size, interpolation=interp)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        ax = axs[0]
        ax.imshow(img)
        ax.set_title("Original Image")
        ax.axis("off")

        # Upsampled image
        ax = axs[1]
        ax.imshow(resampled)
        ax.set_title(title)
        ax.axis("off")

        plt.tight_layout()
        plt.show()

    return resampled

#%% Mask the region out of ROI
def create_mask_ROI(img, ROI):
    y0,y1,x0,x1 = ROI
    mask = np.zeros(img.shape, dtype=bool)
    mask[y0:y1,x0:x1] = True
    masked_img = img*mask
    return masked_img

#%% Apply the existing transform (in .mat) to an image
def transforminmat2img(transform_path, moving_img, fixed_img=None,plot=True):
    transform = ants.read_transform(transform_path)
    moving_img = ants.from_numpy(moving_img)
    if fixed_img is not None:
        fixed_img = ants.from_numpy(fixed_img)
    transformed_img = transform.apply_to_image(moving_img,reference=fixed_img)
    if plot:
        if fixed_img is not None:
            plt.imshow(fixed_img.numpy(), cmap='gray')
            plt.imshow(transformed_img.numpy(),alpha=0.5)
        else:
            plt.imshow(transformed_img.numpy())
        plt.tight_layout()
        plt.show()
    return transformed_img.numpy()

#%% register hyperspectral datacube by iterating over each slice along the signal dimension
def transform2hsdata(fixed_img, moving_hsdata, transform, interpolator='nearestNeighbor', transform_type='fwdtransforms',
                     rewrite=False,copy_obj=False,
                     crop_to_overlap_flag=False, bbox=None, overlap_threshold=0,
                     **kwargs):
    """
    Apply the transformation acquired from the registration to the entire hyperspectral dataset
    :param fixed_img (np.ndarray): the fixed image
    :param moving_data (Lumispectrum): the moving hyperspectral image dataset read by hyperspy
    :param transform (dict): the transformation matrix
    :param interpolator (str): the interpolator used for resampling the image, default is 'nearestNeighbor'
    :param transform_type (str): the type of the transformation, default is 'fwdtransforms'
    :param rewrite (bool): whether to rewrite the data in the moving_hsdata, default is False
    :param copy_obj (bool): whether to return a copy of the moving_hsdata with updated data, default is False
    :param crop_to_overlap_flag (bool): whether to crop the registered data to the overlapping region, default is False
    :param bbox (tuple): the bounding box of the overlapping region, default is None
    :param overlap_threshold (float): the intensity threshold for overlap, default is 0
    :return: warped_hsdata (Lumispectrum): the registered hyperspectral image dataset if rewrite or copy_obj is True, or
    warped_data (np.ndarray): the registered hyperspectral image dataset if both rewrite and copy_obj are False
    """
    warped_data = np.zeros(moving_hsdata.data.shape)
    for i in range(moving_hsdata.data.shape[2]):
        fixed_data = ants.from_numpy(fixed_img)
        moving_data = ants.from_numpy(moving_hsdata.data[:,:,i])
        warped_data[:,:,i] = ants.apply_transforms(fixed=fixed_data, moving=moving_data, transformlist=transform[transform_type],
                                                   interpolator=interpolator,
                                                   **kwargs).numpy()
    # Crop if requested
    if crop_to_overlap_flag:
        if bbox is None:
            print("⚠️ No bounding box provided for cropping. Computing overlap from data.")
            _, bbox = return_overlap(warped_data.sum(axis=2), threshold=overlap_threshold, plot=False)
        warped_data = crop_to_overlap(warped_data, bbox)

    if rewrite:
        moving_hsdata.data = warped_data
        return moving_hsdata
    elif copy_obj:
        warped_hsdata = copy.deepcopy(moving_hsdata)
        warped_hsdata.data = warped_data
        return warped_hsdata
    else:
        return warped_data
