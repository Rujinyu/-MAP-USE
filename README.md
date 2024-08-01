# MAP-USE Model (last updated: 2024-08-01)

This repository contains python codes related MAP-USE Model.  This CEUS video-based model potentially enables real-time scanning for predicting MVI while visualizing the tumor immune microenvironment.The training of the model was completed on NVIDIA GeForce GTX 3090.
- The data required to build the model **dataset.py**
-	Model architecture **model.py**
- Training model **train.py**
- Evaluating model **validate.py**
- Some configuration information **config.py**
- Model attention heatmap visualization **cam.py**
- Conda environment **environment.yml**

**To run the codes ensure you have the necessary prerequisites:**
-	Python installed on your system (version 3.7 or above).
- Using instruction:
`conda env create -f environment.yml -n myenv`


