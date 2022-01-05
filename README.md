# visanalysis
Analysis environment for visprotocol/flystim imaging experiments
Max Turner
mhturner@stanford.edu

INSTALLATION:
cd to top-level visanalysis/
pip install -e .

---

**Using the Data GUI** 


![ss_annotated](https://user-images.githubusercontent.com/9029384/148145301-2e75b87b-8f6f-4ee5-ae05-c5e8515a27d7.jpg)


1) run DataGUI.py from the command line
2) Load experiment (HDF5) file
3) Select data directory. This should be a directory that contains all of the image series files (.nii or .tif) you want to examine for this experiment
4) Select a group in the Data tree browser to examine fly metadata or run parameters. Selecting an epoch run series will load any associated image series. If none can be found in the data directory you have selected, it will prompt you to select the image file associated with this series
