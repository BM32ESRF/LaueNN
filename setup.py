import os
import pathlib
import setuptools
from setuptools import find_packages, setup
 
with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    long_description = readme.read()

setuptools.setup(
    name="lauetoolsnn",
    use_scm_version=True,

    #version="3.0.79", ##Automatic versioning with github tag

    author="Ravi raj purohit PURUSHOTTAM RAJ PUROHIT",
    
    author_email="purushot@esrf.fr",
    
    description="LaueNN- neural network training and prediction routine to index single and polycrystalline Laue diffraction patterns",

    long_description=long_description,
    
    long_description_content_type="text/markdown",
    
    include_package_data=True,
    
    packages=find_packages(),
    
    url="https://github.com/BM32ESRF/LaueNN",
    
    setup_requires=['setuptools_scm', 'matplotlib', 'Keras', 'scipy','numpy', 'h5py', 'tensorflow', 'PyQt5', 'scikit-learn', 'fabio', 'networkx', 'scikit-image', 'tqdm'],
    install_requires=['matplotlib>=3.4.2', 'Keras>=2.7.0', 'scipy>=1.7.0','numpy>=1.18.5', 'h5py>=3.1', 'tensorflow>=2.7.0', 'PyQt5>=5.9', 'scikit-learn>=0.24.2', 'fabio>=0.11.0', 'networkx>=2.6.3', 'scikit-image>=0.18.0','tqdm>=4.60.0'],


    entry_points={
                 "console_scripts": ["lauetoolsnn=lauetoolsnn.lauetoolsneuralnetwork:start",
                                     "lauenn=lauetoolsnn.lauetoolsneuralnetwork:start", 
                                     "lauenn_addmat=lauetoolsnn.util_scripts.add_material:start",
                                     "lauenn_mat=lauetoolsnn.util_scripts.add_material:querymat",
                                     "lauenn_maxhkl=lauetoolsnn.util_scripts.add_material:query_hklmax",
                                     "lauenn_pymatgen=lauetoolsnn.util_scripts.add_material:pymatgen_query",
                                     "lauenn_example=lauetoolsnn.util_scripts.add_material:example_scripts"]
                 },
                 
    classifiers=[
                    "Programming Language :: Python :: 3.7",
                    "Programming Language :: Python :: 3.8",
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10",
                    "Topic :: Scientific/Engineering :: Physics",
                    "Intended Audience :: Science/Research",
                    "Development Status :: 5 - Production/Stable",
                    "License :: OSI Approved :: MIT License "
                ],
                
    python_requires='>=3.7, <=3.10',
    # >=3.7 is required becquse of PyQt5
)
