============
Known Issues
============
So far, there is a issue with ``H5py`` and ``HDF5`` version mismatch in the windows installation with conda. If error with H5py version mismatch exist after conda installation, please try ``pip install lauetoolsnn`` on windows as this should not have this problem. The other possibility is to install the H5py with pip before or after installing lauetoolsnn with conda.

Additionally, if there are issues related to ``PyQt5`` upon calling ``lauenn`` from terminal. Possible workaround is to install PyQt5's appropriate version for the python version and then additionally instally the ``PyQtWebEngine`` will most probably solve the issue.
