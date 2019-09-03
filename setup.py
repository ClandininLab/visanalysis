from setuptools import setup

setup(
    name='visanalysis',
    version='0.1.0',
    description='Analysis environment for flystim experiments',
    url='https://github.com/ClandininLab/visanalysis',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['visanalysis'],
    install_requires=[
        'PyQT5',
        'numpy',
        'h5py',
        'scipy',
        'pandas',
        'scikit-image',
        'thunder-registration',
        'seaborn',
        'pyqtgraph',
        'pyyaml'],
    include_package_data=True,
    zip_safe=False,
)
