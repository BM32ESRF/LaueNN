============
Installation
============
Lauetoolsnn can be installed either via PYPI usiing the following command in terminal (this installs all dependencies automatically):

`PYPI repository <https://pypi.org/project/lauetoolsnn/>`_

`Anaconda repository <https://anaconda.org/bm32esrf/lauetoolsnn>`_

.. code-block:: console

   pip install lauetoolsnn
   conda install -c bm32esrf lauetoolsnn -c conda-forge


For macOS user, please use the Anaconda installation to avoid build errors or can be compiled and installed locally via the setup.py file. Download the Github repository and type the following in terminal. In this case, the dependencies has to be installed manually. The latest version of each dependency works as of (01/04/2022).

.. code-block:: console

   git clone https://github.com/ravipurohit1991/lauetoolsnn.git
   cd luetoolsnn
   python setup.py install

Naturally, you can also install the lauetoolsnn package directly with the ANacondad Navigator. On the Anaconda Navigator, once you have created your own environment with ``python>=3.7``\; configure ``channels`` using the channels button and add ``conda-forge`` and ``bm32esrf``. After updating the index, you should have lauetoolsnn package accessible via the search bar. 

See `Procedure for Config file generation <https://github.com/ravipurohit1991/lauetoolsnn/blob/main/presentations/procedure_usage_lauetoolsnn.pdf>`_ for installation and how to write the configuration file to be used with GUI.
This project is also hosted on `sourceforge <https://lauetoolsnn.sourceforge.io>`_.
