# Project Title
EEE511 Team 2 FinalProject

### Prerequisites:
Python 2.7 or 3.4 or 3.5 or 3.6 (Noted that Python 2.7 is not support
by TensorFlow on windows)
Pip(pip3)
pip install numpy
pip install pandas
pip install catboost
pip install sklearn
pip install matplotlib
pip install lightgmb
pip install xgboost
pip install tqdm
Tensorflow:
Pip install –-user –-upgrade tensorflow

Download the dataset from https://drive.google.com/open?id=1rlua_h9GKOOm_4OhROOr447yYakJqFSP
Put this 'data' folder (Inside the folder there should be six csv file) beside the python file. 

### Installing
1. Run the EEE511_FinalProject.py 
2. Wait a few minutes for the data preprocessing.
3. There will be 4 File generated which is:
	a. LightGBMResult_submission_nofold.csv : The predict result from LightGBM
	b. XGBResult_submission_nofold.csv : The predict result from XGBoost
	c. CatBoostResult_submission_nofold.csv : The predict result from CatBoost
	d. StackingResult.csv : The predict result from the stacking model. 

These csv file can be upload to https://www.kaggle.com/c/kkbox-music-recommendation-challenge to see the AUC score.

You have to login to submit the result to kaggle.