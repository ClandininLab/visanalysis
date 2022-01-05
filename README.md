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
5) You may delete groups at any level of the tree you would like (Fly, image series, roi, etc) using the "Delete selected group" button
6) Once you've loaded an image file associated with a series, you can use the Image & roi canvas to draw ROIs on the image.

    a) Left (primary) click to draw a single ROI. Right (secondary) click to draw a ROI that you can draw in separate, discontiguous segments (including across z levels, by changing the z slider at the bottom of the canvas). For discontiguous ROIs, hit your ENTER key to finish the ROI.
    
    b) Once you have drawn a ROI, the response trace will appear. You can change what appears in the trace window using the dropdown menu to, for example, examine a simple trial average. 
    
    c) You can add a number of ROIs to a roi set. Examine different ROIs within your roi set by moving the ROI set slider.
    
    d) Once you are happy with your ROI set, enter a roi_set_name and click "Save ROIs." The roi information for this roi set, including the paths you drew, a mask for the rois, and the response trace(s) will be saved directly to the HDF5 file and will now appear in the Data Tree Browser.
    
7) You can now access the ROI data directly from the HDF5 file and no longer need the image or metadata files (unless you want to draw more ROIs). See the example script visanalysis/analysis/example_plotting.py for some inspiration on how to do this.


*Notes & Tips for using the DataGUI*
- Always keep a backup copy of your raw data file as it was generated during the experiment. You can always go through the steps of re-attaching metadata or re-drawing ROIs if necessary.
- The GUI will look for image files in the data directory with the following format: TSeries-yyyymmdd-00n.suffix or TSeries-yyyymmdd-00n_reg.suffix (for registered / motion corrected series). If it doesn't find a matching file, it will prompt you to select it yourself. After you have done so once, the file name will be saved to the HDF5 file so in the future you shouldn't have to always manually point to the file. If you'd like to change the associated image file for a series, click "Select image data file" to redefine this file.
- Before you do anything with these data files, you need to attach data to your HDF5 file, including imaging metadata and stimulus timing information. This metadata comes from associated metadata files generated during imaging. For Bruker, for example, it requires the bruker .xml file, the photodiode .csv and photodiode .xml. There is a button to do this ("Attach metadata to file") and there are also visanalysis modules & scripts to do it as well.
- Keep an eye on the terminal window running the GUI, as it sometimes prints useful or informative things. Especially when it crashes.

