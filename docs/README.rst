
Lauetoolsnn/ LaueNN
-------------------
An autonomous feed-forward neural network (FFNN) model to predict the HKL in single/multi-grain/multi-phase Laue patterns with high efficiency and accuracy is introduced. 

Laue diffraction indexation (especially Laue images comprising of diffraction signal from several polycrystals/multi phase materials) can be a very tedious and CPU intensive process. To takle this, LaueNN or LauetoolsNN was developed employing the power of neural network to speed up a part of the indexation process. In the `LaueNN_presentation <https://github.com/BM32ESRF/LaueNN/tree/main/presentations/LaueNN_presentation.pdf>`_, several steps of Laue pattern indexation with classical approach is described. We have replaced the most CPU intensive step with the Neural Networks. The step where the Laue indices hkl of each spot os now determined with the Neural networks, alongside the spot hkl index, the neural network also predicts the Material that spot belongs to. This can be useful incase of Laue images comprising of diffraction signal from multi-phases. LaueNN uses the existing modules of Lauetools to generate simulated Laue patterns. The whole workflow and the application of this tool is illustrated in this article `LaueNN: neural-network-based hkl recognition of Laue spots and its application to polycrystalline materials <https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576722004198>`_.


Requirements: (latest version of each libraries accessed on 03/04/2022) 
------------------------------------------------------------------------------------ 
- PyQt5 (GUI)
- matplotlib
- Keras
- tensorflow 
- numpy 
- scipy (scipy transform rotation is used)
- h5py (required for writing neural network model files)
- scikit-learn (required for generating trained model classification reports)
- fabio (used for opening raw Laue tiff images)
- networkx (to be replaced with numpy in the future)
- scikit-image (used for hough based analysis of Laue patterns)
- tqdm (required only for notebook scripts)
- opencv (for LOG based peak search)


Citation
--------------
If you use this software, please cite it using the metadata available in the citation_bibtex.cff file in root.

``Purushottam Raj Purohit, R. R. P., Tardif, S., Castelnau, O., Eymery, J., Guinebretiere, R., Robach, O., Ors, T. & Micha, J.-S. (2022). J. Appl. Cryst. 55, 737-750.``


Support
--------------
Do not hesitate to contact the development team at purushot@esrf.fr or micha@esrf.fr .


Maintainer(s)
--------------
* Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (https://github.com/ravipurohit1991)


License
--------------

The project is licensed under the MIT license.
