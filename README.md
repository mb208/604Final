# 604Final
## Fall 2023 Stats 604 Final Project: Predicting the Weather
The goal of this project is to predict some daily atmospheric measurement up to four days in the future (e.g. if it 12/4/2023 we predict the values for 12/5-12/8). The atmospheric variables we consider are minimum, maximum, and average temperature as well as whether it rains or snows. We consider 4 algorithms: 
* LSTM
* SARIMA
* GAMs 
* XGBoost 

The `notebooks ` directory contains jupyter notebooks performing either hyper-parameter tuning or cross-validation for the LSTM's, GAM, and XGBoost. The `predict` folder contains working predict files for the LSTM and XGBoost models. The `train` folder contains working train files for the LSTM and XGBoost models.

For our final model we used an LSTM. This is available as a docker container. To access the docker container run 
`docker pull mgb208/604grp3:latest`
if using Mac and 
`docker pull mgb208/604grp3:latest-amd64` 
if using linux.


To obtain predictions for the next 4 days run
`docker run mgb208/604grp3:latest predict`.
To retrain the model with the most recently available data run  
`docker run mgb208/604grp3:latest train -v ${PWD}:/home/docker/output` to store the recently trained model in your current directory.