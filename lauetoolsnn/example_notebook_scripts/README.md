# lauetoolsnn notebook scripts 
$$\textcolor{red}{\text{These scripts are not adapted for the new version of LaueNN and may not function properly. Please check out the end-to-end scripts folder.}}$$

In the github repository of the project example notebook scripts are included to build a neural network model for any crystal symmetry as described in this article.  These notebook scripts aims to provide a general tutorial into the complete flow of LaueNN method. 

## Step 1: Generation of training dataset script: https://github.com/ravipurohit1991/lauetoolsnn/blob/main/lauetoolsnn/example_notebook_scripts/Step1_Generation_dataset_LaueNN.ipynb
This notebook script provides information on how the output hkl class of the neural network is build given any crystallographic symmetry. Once defined, the simulated Laue patterns training dataset is generated and saved. 

## Step 2: Training a neural network: https://github.com/ravipurohit1991/lauetoolsnn/blob/main/lauetoolsnn/example_notebook_scripts/Step2_Training_LaueNN.ipynb
This notebook give introduction to defining a neural network architecture. Since the input and output classes are already generated. It is possible to define another neural network architecture also, for example a 1D CNN model that takes in same input and output. Additionally another script is provided that helps the users to play with different hyper parameters of the model architecture to optimize it specifically for their case (Sub-step 2a: Optimize hyper parameters of neural network architecture: https://github.com/ravipurohit1991/lauetoolsnn/blob/main/lauetoolsnn/example_notebook_scripts/Step2a_Optimize_architecture_LaueNN.ipynb)

## Step 3: verify the neural network prediction and indexation of Laue Patterns: https://github.com/ravipurohit1991/lauetoolsnn/blob/main/lauetoolsnn/example_notebook_scripts/Step3a_Generate_simulateLPforPrediction_LaueNN.ipynb
This script describes the steps for predicting Laue spots hkl for a given input and also includes functions that reconstructs orientation matrix and indexes the crystal of the given dataset. A script to generate simulated Laue patterns for prediction is also provided. (Step 3a: Generation of simulated dataset for prediction: https://github.com/ravipurohit1991/lauetoolsnn/blob/main/lauetoolsnn/example_notebook_scripts/Step3_Prediction_LaueNN.ipynb)
The scripts have default values that should allow any user to launch all the scripts in sequence to see the working flow of the method.

