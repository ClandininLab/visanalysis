# visanalysis
Visanalysis is an analysis environment for [visprotocol](github.com/clandininlab/visprotocol) imaging experiments. It is designed for visual stimulation and functional imaging data, but can be the basis for handling experiments with different datatypes and stimulation protocols, as well. The central idea behind the analysis approach is to get data from disparate sources (e.g. different microscopes or other recording devices) into a common datafile format, which can then be accessed and analyzed by a shared set of downstream analyses.


Contact: Max Turner, mhturner@stanford.edu

## Installation
- visanalysis has been tested with Python 3.9 and Python 3.6, on OSX and Linux. It may work on Windows, but it hasn't been tested.

### Dependencies
- PyQT6 is required for the GUI (optional)

### Install visanalysis
It helps to start with a fresh conda environment. But this is not strictly necessary.

`conda create -n visanalysis python=3.9`

`conda activate visanalysis`

1. **GUI-free:** To install the basic, GUI-free version. cd to top-level visanalysis/, where setup.py lives, and run:
  
    `pip install -e .`

2. **GUI:** To be able to use the GUI, run:
    `pip install -e .[gui]` in bash or `pip install -e ".[gui]"` in zsh (note quotes)

## Documentation and examples
- /examples contains some example scripts that you can use to get oriented to visanalysis and to test things out on your machine. The example_data are not included in this repository. You may download example data [here](https://drive.google.com/drive/folders/1oJYcUjXBudPpiCPd4wDlIYoWLE30CURR?usp=sharing)
- Check out the [wiki](https://github.com/ClandininLab/visanalysis/wiki) for more detailed documentation & how-tos



