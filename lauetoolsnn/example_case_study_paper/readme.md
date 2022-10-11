This repository contains the data and prediction results of the neural network model for case studies presented in the article "LaueNN: Neural network based hkl recognition of Laue spots and its application to polycrystalline materials".

The results can be visualized by loading the results.pickle file in the lauetoolsnn GUI --> Live prediction with IPF map button and then file--> load results tab.

Apart from this using a script the results can also be visualized : lauetoolsnn\util_scripts\plots_postprocess.py and lauetoolsnn\util_scripts\unit_cell_relative_plots.py to load the results and plot them.

For IPF maps, the lauetoolsnn\util_scripts\rewrite_mtex.py script will write ".ctf" data which can be opened via matlab based open source MTEX orientation analysis software.
