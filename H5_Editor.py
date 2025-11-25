import tables

#%% extract file paths from a folder
def h5_paths(folder, endswith='.h5', keywords=None):
    """
    Function to extract file paths from a folder based on file extension and keywords
    :param folder: Path to the folder
    :param endswith: File extension to filter (default: '.h5')
    :param keywords: List of keywords that must be present in the filename (default: None)
    :return: List of file paths that match the criteria
    """
    import os
    matched_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Check extension
            if not file.endswith(endswith):
                continue
            # If keywords provided, ensure they all appear in the filename
            if keywords:
                if not all(keyword in file for keyword in keywords):
                    continue
            matched_files.append(os.path.join(root, file))
    return matched_files
#%% save the input h5 dataset(s) to an HDF5 files (pytables)
def save2H5(h5_paths, group_names,filename,filetitle='Combined h5 file', savepath=None):
    """
    Function to save a list of HDF5 files to a new HDF5 file
    :param h5_paths: List of HDF5 file paths
    :param group_names: List of new group names to save from each HDF5 file
    :param filename: Name of the new HDF5 file
    :param filetitle: Title of the new HDF5 file (default: 'Combined h5 file')
    :param savepath: Path to save the new HDF5 file (default: None)
    """
    if savepath is None:
        savepath = ''
        print("Warning: No save path provided, saving to current directory.")
    with tables.open_file(savepath + filename+'.h5', mode="w", title=filetitle) as new_file:
        for i in range(len(h5_paths)):
            with tables.open_file(h5_paths[i], mode='r') as existing_file:
                group_to_copy = existing_file.get_node('/Datas')
                new_file.copy_node(group_to_copy, new_file.root, newname=group_names[i], recursive=True)
                print('The structue of the file:\n', new_file)
                existing_file.close()
    new_file.close()
    print("Data saved to", savepath + filename,' successfully and the file was closed.')
#%% rename groups in an HDF5 file
def node2rename(h5_path, old_name, new_name, loc_node=None):
    """
    Function to rename groups or subgroups in an HDF5 file
    :param h5_path: Path to the HDF5 file
    :param old_name: Old name of the group
    :param new_name: New name of the group
    :param loc_node: Location of the node (default: None, if the node is in the root, i.e., a group;
    otherwise, the name of the group, e.g. '/Group'or'/Group/Subgroup')
    """
    with tables.open_file(h5_path, mode='a') as h5_file:
        if loc_node:
            h5_file.rename_node(loc_node+'/'+old_name, new_name) # rename a subgroup
            print(f'The subgroups in {loc_node}:', h5_file.get_node(loc_node))
        else:
            h5_file.rename_node('/'+old_name, new_name) # rename a group under root
            print('The groups in the file:\n', h5_file.get_node('/'))
    h5_file.close()
    print('The file was closed.')
#%% extract data, wavelength and axes info from an H5 file of hyperspectral data
def data_extract(h5_path, data_loc='/Datas/Data1', metadata_loc=None,
                 wl_attr='Axis1',x_axis_attr='Axis2', y_scale = False, y_axis_attr='Axis3'):
    """
    Function to extract data, signal axis and pixel size from an HDF5 (h5) file
    :param h5_path: Path to the HDF5 file
    :param data_loc: Location of the data in the file (default: '/Datas/Data1')
    :param metadata_loc: Location of the metadata in the file (default: None)
    :param wl_attr: Name of the wavelength axis attribute (default: 'Axis1')
    :param x_axis_attr: Name of the x axis attribute (default: 'Axis2')
    :param y_scale: Boolean to indicate if the y axis needs to be returned (default: False)
    :param y_axis_attr: Name of the y axis attribute (default: 'Axis3')
    :return: data, wavelength_axis, x_px_size, y_px_size (if y_scale is True)
    """
    import numpy as np

    h5_file = tables.open_file(h5_path, mode='r')
    data_node = h5_file.get_node(data_loc)
    data = data_node.read()
    if metadata_loc:
        metadata_node = h5_file.get_node(metadata_loc)
    else:
        metadata_node = data_node

    wavelength_axis = metadata_node.attrs[wl_attr]
    print(f"Wavelength axis unit: {metadata_node.attrs[wl_attr+' Unit'].decode('utf-8')}")
    x_axis = metadata_node.attrs[x_axis_attr]
    print(f"Pixel size in x axis unit: {metadata_node.attrs[x_axis_attr+' Unit'].decode('latin-1')}")

    # check if position axis is uniformly spaced
    def is_equally_spaced_np(arr, tol=0):
        arr = np.asarray(arr)
        diffs = np.diff(arr)
        return np.all(np.abs(diffs - diffs[0]) == tol)
    if is_equally_spaced_np(x_axis):
        x_px_size = x_axis[1] - x_axis[0]
    else:
        x_px_size = x_axis
        print("Warning: The x axis is not equally spaced. The full x axis array is returned.")
    if y_scale:
        y_axis = metadata_node.attrs[y_axis_attr]
        print(f"Pixel size in y axis unit: {metadata_node.attrs[y_axis_attr + ' Unit'].decode('latin-1')}")
        if is_equally_spaced_np(y_axis):
            y_px_size = y_axis[1] - y_axis[0]
        else:
            y_px_size = y_axis
            print("Warning: The y axis is not equally spaced. The full y axis array is returned.")
        h5_file.close()
        return data, wavelength_axis, x_px_size, y_px_size
    else:
        h5_file.close()
        return data, wavelength_axis, x_px_size