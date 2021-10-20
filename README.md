# Interacting-Vehicle-Trajectory-Prediction-with-Convolutional-Recurrent-Neural-Networks

Anticipating the future trajectories of surrounding vehicles is a crucial and challenging task in path planning for autonomy. We propose a novel Convolutional Long Short Term Memory (Conv-LSTM) based neural network architecture to predict the future positions of cars using several seconds of historical driving observations. This consists of three modules: 1) Interaction Learning to capture the effect of surrounding cars, 2) Temporal Learning to identify the dependency on pastmovements and 3) Motion Learning to convert the extracted features from these two modules into future positions. To continuously achieve accurate prediction, we introduce a novel feedback scheme where the current predicted positions of each car are leveraged to update future motion, encapsulating the effect of the surrounding cars. Experiments on two public datasets demonstrate that the proposed methodology can match or outperform the state-of-the-art methods for long-term trajectory prediction.

**Steps to run the code:**

1. Run the LoadDataV12 to generate the OGM Maps and save it in a folder. This will produce both the train and validation data. Before running check and provide the file path for the NGSIM csv data file. 
2. Then run the TrainValLSTM to train and save the model. This time pass the folder created in the previous step as the input for training. provide the parent folder which contains both the training and validation data folders. 
3. Then run the PredictOGMV4 to predict the future trajectory. This time also pass the main NGSIM csv data file path. This code will create the intermediate OGM files right before prediction and then save it in some intermediate folder and then do the prediction and save the result/RMSE error in some internal list, then clear up the intermediate folder and then will do the same for the next sample and so on. 


**If you are using this code please cite the following paper:**

Mukherjee, Saptarshi, Sen Wang, and Andrew Wallace. "Interacting vehicle trajectory prediction with convolutional recurrent neural networks." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.
