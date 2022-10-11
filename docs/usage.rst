========
Usage
========

After installation, accessing the GUI package of LaueNN is simple. Once needs to open the terminal, activate the conda environment in which LauetoolsNN was installaed and simply type the following commands:

Query if the material you want to index exist in LaueNN or not with 
``lauenn_mat -n mat_key`` : If your material is silicon, replace mat_key with Si. This should return the lattice parameters (a, b, c, alpha, beta, gamma) and general reflection condition for this material in the LaueNN database.


If your material does not exists or if you need to create a material with different lattice parameters and reflection condition. One can do that with the following:
``lauenn_addmat -n user_name -l a b c alpha beta gamma -e spage_group_number`` : you can create your own material using the following command, replace the lattice parameters for (a,b,c,alpha,beta,gamma) and space group number of your crystal.

Once the material is added, now we can do the training and prediction with the lauetoolsnn GUI.
``lauetoolsnn`` or ``lauenn``: This opens the GUI of LauetoolsNN, from which Training and Prediction of Laue data can be carried out. Only a config file is required to start the autonomous analysis. To see how to generate a config file adapted for your case, please refer to `config file creation tutorial <../latest/GUi_functions/config_file.html>`_.
