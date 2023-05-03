# Future Frame Prediction and Segmentation - Final Project Deep Learning (Spring 2023)

Goal:
Problem Statement - Using the first 11 frames of a video predict the segmentation mask of the last (22nd) frame.

### to create Conda Environment 

the following command can be used to create the conda environment and install all necessary dependencies:
```bash
conda env create -f environment.yml
source activate projectDL
```

### Executing Experiments

##### Future frame Prediction
Generative Adversarial NEtwork (GAN) with a ConvLSTM generator and a simple linear discriminator was used for future frame prediction.

To run the pipeline and train the generator network to predict the 22nd frame, the following command can be executed - 

```
python frame_pred/src/main_hpc.py --cfg=config_hpc.json
```

##### Segmentation
Segmentation is performed using a U-Net model. 

to run the pipeline and generate and train the segmentation model, the following command can be executed -

```
python segmentation/segmentation.py
```
### Inference
to connect both the future frame prediction model and the Segmentation model in order to generate the masks of the 22nd frame of a video (given only 11 frames as input), the following command can be executed : 

```
python infer.py
```

** Notes **

This project was part of the coursework for the Deep Learning (spring 2023) course at NYU.

Contributors:
* Charvi Gupta (cg4177@nyu.edu)
* Anoushka Gupta (ag8733@nyu.edu)
* Anisha Bhatnagar (ab10945@nyu.edu)
