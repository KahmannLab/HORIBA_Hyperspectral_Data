# HORIBA_Hyperspectral_Data
Here are modules and examples for HORIBA hyperspectral data analysis, which work for hyperspectral data visulization, simple machine learning task, i.e., principal component analysis (PCA) and image registration.

## Requirements
- hyperspy (if working with .xml files exported from Labspec)
- tables (if working with .h5 files)
- numpy
- matplotlib
- matplotlib-scalebar
- scipy
- scikit-learn (for PCA)
- ANTsPy (for image registration)
- numba (for processes parallelization)

## Usages
- **If hyperspy is used to read xml file, please replace the file named as _api.py (venv -> rsciio -> jobinyvon) by the file with the same name provided in this repository.** The modified file figures out the incompatible parameters and the unmatched row length in the xml file exported from our Labspec6 software.
- Save the HORIBA_HyperspectralData module and the example file into the same folder.
- The examples of those modules are shown in Example_HORIBA_HSData.py.
