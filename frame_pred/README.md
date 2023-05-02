## Future Frame Prediction Model


### Model Architecture
An Adversarial Spatio-Temporal Convolutional LSTM architecture to predict the future frames


### Files
`src/config_local.json` - Config file containing input dimensions, number of future frames to predict, paths, and model training specifications. Suggested to create a different config file for each run

`src/logs` - Training logs

`src/board` - Tensorboard logs

`model` -  Saved model

`results` - Visualize model input and output at every log frequency steps. Contains the ground truth and predicted frames for the train dataset and test dataset


### Model Input
Torch.Tensor object of shape (batch_size, num_future_frames, n_channels, image_height, image_width) containing past frames from timestamp 0 to num_future_frames

### Output
Torch.Tensor object of shape (batch_size, 2 * num_future_frames, n_channels, image_height, image_width) containing predicted future frames from timestamp 0 to 2 * num_future_frames

### Training Time and Configuration
Trained for 2000 train and validated on 1000 videos, on NYU HPC on 4 V100 GPU using Data Parallel, for 50 epochs in ~20 hours

### To run on local - 
Create the conda environment. Use `src/config_local.json` and run `python main.py --cfg=config_local.json` from `src` directory


### To run on NYU HPC - 
Create the conda environment. Use `src/config_hpc.json` and run `python main_hpc.py --cfg=config_hpc.json` from `src` directory. Suggested slurm parameters present in `run-main-objects.SBATCH` file.


### Evaluation Metric
Generator Loss - adversarial_loss + 4 * l1_l2_loss
Discriminator Loss - adversarial_loss


## References - 

> Base code: https://github.com/vineeths96/Video-Frame-Prediction

> Data Parallel implementation in Pytorch: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
