from setuptools import setup

setup(
    name='visanalysis',
    version='2.0.0',
    description='Analysis environment for visprotocol experiments',
    url='https://github.com/ClandininLab/visanalysis',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['visanalysis'],
    install_requires=['PyQT6',
                      'numpy',
                      'h5py',
                      'scipy',
                      'pandas',
                      'scikit-image',
                      'matplotlib',
                      'npTDMS',
                      'nibabel',
                      'psutil'],
    extras_require={'gui':  ["PyQT6"]},
    include_package_data=True,
    zip_safe=False,
)
