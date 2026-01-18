import tables
import os
#%% extract file paths from a folder
def h5_paths(folder, endswith='.h5', keywords=None):
    """
    Function to extract file paths from a folder based on file extension and keywords
    :param folder: Path to the folder
    :param endswith: File extension to filter (default: '.h5')
    :param keywords: List of keywords that must be present in the filename (default: None)
    :return: List of file paths that match the criteria
    """
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
def normalize_path(path: str):
    return os.path.abspath(os.path.normpath(path))
# print the data tree
def print_tree(h5_path):
    h5_path = normalize_path(h5_path)
    with tables.open_file(h5_path, mode='r') as h5_file:
        print(h5_file)

def save2H5(h5_paths, group_names, filename, filetitle='Combined h5 file', savepath=None):

    if savepath is None:
        savepath = ''
        print("Warning: No save path provided, saving to current directory.")

    out_path = normalize_path(os.path.join(savepath, filename + '.h5'))

    with tables.open_file(out_path, mode="w", title=filetitle) as new_file:

        for i in range(len(h5_paths)):
            src_path = normalize_path(h5_paths[i])

            with tables.open_file(src_path, mode='r') as existing_file:
                group_to_copy = existing_file.get_node('/Datas')
                new_file.copy_node(group_to_copy, new_file.root,
                                   newname=group_names[i], recursive=True)

            # DO NOT call existing_file.close()
            # DO NOT call new_file.close()

    print("Data saved to:", out_path)
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
    h5_path = normalize_path(h5_path)
    with tables.open_file(h5_path, mode='a') as h5_file:
        if loc_node:
            h5_file.rename_node(loc_node+'/'+old_name, new_name) # rename a subgroup
            print(f'The subgroups in {loc_node}:', h5_file.get_node(loc_node))
        else:
            h5_file.rename_node('/'+old_name, new_name) # rename a group under root
            print('The groups in the file:\n', h5_file.get_node('/'))
    print('The file was closed.')

def attribute2rename(h5_path, node_path, old_name, new_name):
    """
    Rename an attribute on any HDF5 node using PyTables.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    node_path : str
        Path to the node (e.g. '/', '/group1', '/group1/datasetA').
    old_name : str
        Existing attribute name.
    new_name : str
        New attribute name.
    """
    h5_path = normalize_path(h5_path)
    with tables.open_file(h5_path, mode='a') as h5:  # must be writable
        node = h5.get_node(node_path)
        attrs = node._v_attrs

        # Check if the old attribute exists
        if old_name not in attrs._v_attrnames:
            raise KeyError(f"Attribute '{old_name}' does not exist on node '{node_path}'")

        # If new name already exists, refuse to overwrite
        if new_name in attrs._v_attrnames:
            raise KeyError(f"Attribute '{new_name}' already exists on node '{node_path}'")

        # Copy value
        attrs._f_rename(old_name, new_name)
#%% extract data, wavelength and axes info from an H5 file of hyperspectral data
def data_extract(h5_path, data_loc='/Datas/Data1', metadata_loc=None,
                 wl_attr='Axis1',x_axis_attr='Axis2',decode=True, y_scale = False, y_axis_attr='Axis3'):
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
    h5_path = normalize_path(h5_path)
    h5_file = tables.open_file(h5_path, mode='r')
    data_node = h5_file.get_node(data_loc)
    data = data_node.read()
    if metadata_loc:
        metadata_node = h5_file.get_node(metadata_loc)
    else:
        metadata_node = data_node

    wavelength_axis = metadata_node.attrs[wl_attr]
    if decode:
        wavelength_unit = metadata_node.attrs[wl_attr+' Unit'].decode('latin-1')
    else:
        wavelength_unit = metadata_node.attrs[wl_attr+' Unit']
    wavelength_info = {'Wavelength':wavelength_axis, 'Unit':wavelength_unit}
    #(f"Wavelength axis unit: {metadata_node.attrs[wl_attr+' Unit'].decode('latin-1')}")
    x_axis = metadata_node.attrs[x_axis_attr]
    if decode:
        x_axis_unit = metadata_node.attrs[x_axis_attr+' Unit'].decode('latin-1')
    else:
        x_axis_unit = metadata_node.attrs[x_axis_attr+' Unit']
    #print(f"Pixel size in x axis unit: {metadata_node.attrs[x_axis_attr+' Unit'].decode('latin-1')}")

    # check if position axis is uniformly spaced
    def is_equally_spaced_np(arr, tol=0):
        arr = np.asarray(arr)
        diffs = np.diff(arr)
        return np.all(np.abs(diffs - diffs[0]) == tol)
    if is_equally_spaced_np(x_axis):
        x_px_size = x_axis[1] - x_axis[0]
        x_axis_info = {'Pixel size': x_px_size,'Unit': x_axis_unit}
    else:
        print("Warning: The x axis is not equally spaced. The full x axis array is returned.")
        x_axis_info = {'X axis': x_axis, 'Unit': x_axis_unit}
    if y_scale:
        y_axis = metadata_node.attrs[y_axis_attr]
        #print(f"Pixel size in y axis unit: {metadata_node.attrs[y_axis_attr + ' Unit'].decode('latin-1')}")
        if decode:
            y_axis_unit = metadata_node.attrs[y_axis_attr + ' Unit'].decode('latin-1')
        else:
            y_axis_unit = metadata_node.attrs[y_axis_attr + ' Unit']
        if is_equally_spaced_np(y_axis):
            y_px_size = y_axis[1] - y_axis[0]
            y_axis_info = {'Pixel size': y_px_size,'Unit': y_axis_unit}
        else:
            print("Warning: The y axis is not equally spaced. The full y axis array is returned.")
            y_axis_info = {'Y axis': y_axis, 'Unit': y_axis_unit}
        h5_file.close()
        return data, wavelength_info, x_axis_info, y_axis_info
    else:
        h5_file.close()
        return data, wavelength_info, x_axis_info

def data_extract_temp(h5_path, data_loc='/Confocal PL_p2/Confocal PL_p2',metadata_loc=None,
                      wl_attr='Wavelength',pixel_axis_attr='Pixel size'):
    h5_path = normalize_path(h5_path)
    h5_file = tables.open_file(h5_path, mode='r')
    data_node = h5_file.get_node(data_loc)
    data = data_node.read()
    if metadata_loc:
        metadata_node = h5_file.get_node(metadata_loc)
    else:
        metadata_node = data_node

    wavelength_axis = metadata_node.attrs[wl_attr]
    wavelength_unit = metadata_node.attrs[wl_attr+' Unit']
    pixel_size = metadata_node.attrs[pixel_axis_attr]
    pixel_unit = metadata_node.attrs[pixel_axis_attr+' Unit']
    wavelength_info = {'Wavelength':wavelength_axis, 'Unit':wavelength_unit}
    pixel_info = {'Pixel size': pixel_size,'Unit': pixel_unit}
    h5_file.close()
    return data, wavelength_info,pixel_info
#%% Save groups (as dictionary) to a h5 file
def save_groups_h5(h5_path, groups):
    """
    Save multiple groups and datasets into an HDF5 file using PyTables.
    If the file exists, it is opened in append mode; otherwise, a new file is created.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    groups : dict
        Dictionary in the form:
        {
            "group_name1": {"dataset_name": array-like, ...},
            "group_name2": {...},
            ...
        }
    """
    h5_path = normalize_path(h5_path)
    file_mode = 'a' if os.path.exists(h5_path) else 'w'

    with tables.open_file(h5_path, mode=file_mode) as h5:

        for group_name, datasets in groups.items():

            # Create group if missing
            if not hasattr(h5.root, group_name):
                group = h5.create_group("/", group_name)
            else:
                group = h5.get_node(f"/{group_name}")

            # Write datasets
            for ds_name, data in datasets.items():

                if hasattr(group, ds_name):
                    h5.remove_node(group, ds_name)

                h5.create_array(group, ds_name, obj=data)

    print("Saved:", h5_path)
#%% add attributes (as dictionary) to the h5 file
def attributes2add(h5_path, node_path, attributes):
    """
    Add attributes to any node (file, group, or dataset) inside an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    node_path : str
        HDF5 node path, e.g.:
        "/"                 → root
        "/group1"           → a group
        "/group1/datasetA"  → a dataset/array/table
    attributes : dict
        Dictionary of attribute_name : attribute_value
    """
    h5_path = normalize_path(h5_path)
    with tables.open_file(h5_path, mode='a') as h5:
        node = h5.get_node(node_path)

        for attr_name, attr_value in attributes.items():
            setattr(node._v_attrs, attr_name, attr_value)

#%% add a new array to an existing group in the h5 file
def array2add(h5_path, group_path, array_name, data, *, overwrite=False):
    """
    Add a new array to an existing group in an HDF5 file using PyTables.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    group_path : str
        Path to the target group (e.g. '/group1/subgroup').
    array_name : str
        Name of the array to create.
    data : array-like
        Data to store.
    overwrite : bool, optional
        If True, overwrite an existing array with the same name.
        Default is False.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file does not exist.
    KeyError
        If the group does not exist.
    RuntimeError
        If the array already exists and overwrite=False.
    """

    # --- Normalize path (VERY important on Windows) ---
    h5_path = normalize_path(h5_path)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with tables.open_file(h5_path, mode="a") as h5:

        # --- Get target group ---
        if not h5.__contains__(group_path):
            raise KeyError(f"Group does not exist: {group_path}")

        group = h5.get_node(group_path)

        # --- Check if array already exists ---
        if hasattr(group, array_name):
            if not overwrite:
                raise RuntimeError(
                    f"Array '{array_name}' already exists in '{group_path}'. "
                    "Use overwrite=True to replace it."
                )
            h5.remove_node(group, array_name)

        # --- Create array ---
        h5.create_array(group, array_name, obj=data)

    # File is GUARANTEED closed here

#%% spectroscopy data extraction
def spc_extract(h5_path, data_loc='/Datas/Data1', metadata_loc=None):
    """
    Function to extract data, signal axis and pixel size from an HDF5 (h5) file
    :param h5_path: Path to the HDF5 file
    :param data_loc: Location of the data in the file (default: '/Datas/Data1')
    :param metadata_loc: Location of the metadata in the file (default: None)
    :return: data, wavelength_axis
    """

    h5_path = normalize_path(h5_path)
    h5_file = tables.open_file(h5_path, mode='r')
    data_node = h5_file.get_node(data_loc)
    data = data_node.read()
    if metadata_loc:
        metadata_node = h5_file.get_node(metadata_loc)
    else:
        metadata_node = data_node

    wavelength = metadata_node.attrs['Axis1']
    wavelength_unit = metadata_node.attrs['Axis1 Unit'].decode('latin-1')
    wavelength_axis = {'Wavelength':wavelength, 'Unit':wavelength_unit}
    h5_file.close()
    return data, wavelength_axis