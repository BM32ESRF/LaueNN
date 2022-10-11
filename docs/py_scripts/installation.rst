========================
Python scripts
========================
The python script, jupyter-notebooks and GUI deals only with maximum of two different material per model. In case if you need more than two material, please refer the `multi-material LaueNN code <https://lauetoolsnn.sourceforge.io>`_. The multi-material code can be used with 'N' material. It has been tested with N=5 different material.

An example script to use LaueNN with python script is presented below.

There are three basic steps to launch the LaueNN module. Step 1 and Step 2 need to be run only once per material/case.

#. 
   First step involves defining the material directly in the input dictionary, this will create the training dataset and ground truth class of hkls to be used for Neural network training:

   .. literalinclude:: Step1_Generation_dataset_LaueNN.py


#. 
   Second step involves training the neural network on the data generated in step 1:

   .. literalinclude:: Step2_Training_LaueNN.py


#. 
   Third step involves prediction and saving the results:

   .. literalinclude:: Step3_Prediction_LaueNN.py

