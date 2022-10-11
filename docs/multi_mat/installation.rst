========================
Multi Mat LaueNN scripts
========================
In this repository we have included few scripts that will show the capability of the LaueNN to detect many phases (here 4 phases). The scripts can be modified to include more or less number of materials to detect. 

An example script to use LaueNN with python script is presented below.

There are three basic steps to launch the LaueNN module. Step 1 and Step 2 need to be run only once per material/case.

#. 
   Generation of training dataset and training the neural network script:

   .. literalinclude:: multi_mat_generation_training_LaueNN.py


#. 
   Generate simulated Laue Image of multi-phase material:

   .. literalinclude:: multi_mat_simulatedata_LaueNN.py


#. 
   verify the neural network prediction and indexation of multi-phase Laue Images:

   .. literalinclude:: multi_mat_verifyPrediction_LaueNN.py


#. 
   Multi mat is also available with prediction window in GUI:

   .. literalinclude:: multi_mat_Prediction_LaueNN_interfaceGUI.py