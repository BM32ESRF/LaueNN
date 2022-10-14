========================
Example case with GUI
========================

Two example case studies are included in the lauetoolsnn\examples folder.
Run the GUI by either launching directly from the terminal using the 'lauetoolsnn' command or by running it locally with python lauetoolsneuralnetwork.py command.

First step is to load the config.txt from the example folder, it sets all the values of the GUI to the case study.
In the GUI: 
- Step1: File --> load config . Select the config file from the example directory.

- Step1a: If config file is not available, one can set parameters in the configure parameters window directly.

- Step2: Press the configure parameters button and press Accept button at the end (the values are loaded from the config file).

- Step3: Press Generate Training dataset button. This will generate the training and validation dataset for neural network.

- Step4: Press Train Neural network button. This will start the training process and once finished will save the trained model.

- Step5: Press the Live prediction with IPF map to start the prediction on predefined experimental dataset. Example datafile is included in the examples folder.

- Step6: Once analyzed, the results can be saved using the save results button.

In addition, all the above mentioned steps can be done without the GUI and are detailed in the lauetoolsnn\example_notebook_scripts folder.
Jupyter notebook scripts are provided to run all the steps sequentially.

The indexed orientation matrix is also written in ".ctf" format, which can then be opened with channel 5 Aztec or MTEX software to do post processing related to orientations analysis. MTEX post processing script is also included in the lauetoolsnn\util_script\MTEX_plot.m
