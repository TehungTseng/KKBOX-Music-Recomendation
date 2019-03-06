import numpy as np
import pandas as pd
import NewPrePorcessing as NP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.linear_model import LogisticRegression

md = NP.NewPreProcessing(1)

train, target, real_test_data,predictBase = md.preprocessing()   # you can specify how much data amount you want to use here.

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=7)

#XGboost datatype transform

d_train_xgb = xgb.DMatrix(X_train, label=y_train)
d_valid_xgb = xgb.DMatrix(X_test, label=y_test) 
d_test_xgb = xgb.DMatrix(real_test_data)


watchlist_xgb = [(d_train_xgb, 'train'), (d_valid_xgb, 'valid')]

#lightGBM datatype transform

d_train_lgb = lgb.Dataset(X_train, label=y_train)
d_valid_lgb = lgb.Dataset(X_test, label=y_test) 
d_test_lgb = lgb.Dataset(real_test_data)

valid_sets_lgb =[d_valid_lgb]

#CatBoost datatype transform

pool = cat.Pool(X_train, y_train)
validate_pool = cat.Pool(X_test, y_test)
test_pool = cat.Pool(real_test_data)


# Train model, evaluate and make predictions

params_cat = {
    'iterations': 150,
    'learning_rate': 0.1,
    'eval_metric': 'AUC',
    'random_seed': 42,
    'use_best_model': False
}

params_xgb = {
    'objective' : 'binary:logistic',
    'eta' : 0.7,
    'max_depth' : 8,
    'silent' : 1,
    'eval_metric': 'auc',
    }
    # score :0.59920 @ 0.01 , split = 0.9, random_state=7
    # score :0.59981 @ 0.01 , split = 0.8, random_state=7
    # score :0.59556 @ 0.01 , split = 0.7, random_state=7
params_lgb = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.1 ,
    'verbose': 0,
    'num_leaves': 108,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 128,
    'max_depth': 10,
    'num_rounds': 200,
    } #score :0.62228 @ 0.01 , split = 0.9, random_state=7
      #score :0.62213 @ 0.01 , split = 0.8, random_state=7
      #score :0.62113 @ 0.01 , split = 0.7, random_state=7
model_xgb = xgb.train(params_xgb, d_train_xgb, 105, watchlist_xgb, early_stopping_rounds=20, maximize=True, verbose_eval=10)

model_lgb = lgb.train(params_lgb, d_train_lgb, 100, valid_sets_lgb)

model_cat = cat.train(pool, params = params_cat,logging_level='Verbose')

p_train_xgb = model_xgb.predict(d_valid_xgb)
p_train_lgb = model_lgb.predict(X_test)
p_train_cat = model_cat.predict(X_test)

p_test_xgb = model_xgb.predict(d_test_xgb)
p_test_lgb = model_lgb.predict(real_test_data)
p_test_cat = model_cat.predict(real_test_data)

final_train = p_train_xgb.reshape(-1, 1)
final_test = p_test_xgb.reshape(-1, 1)
final_train = np.concatenate((final_train,p_train_lgb.reshape(-1, 1)), axis=1)
final_test = np.concatenate((final_test,p_test_lgb.reshape(-1, 1)), axis=1)
final_train = np.concatenate((final_train,p_train_cat.reshape(-1, 1)), axis=1)
final_test = np.concatenate((final_test,p_test_cat.reshape(-1, 1)), axis=1)

logistic_regression = LogisticRegression()
        
logistic_regression.fit(final_train.reshape(-1,3),np.array(y_test).reshape(-1,1))

        #predict_labels = logistic_regression.predict_proba(np.array(self.final_test).reshape(-1,len(self.models)))
predict_final = logistic_regression.predict_proba(np.array(final_test).reshape(-1,3))

# Prepare submission
print ("preparing submission........")

lgbResult = np.concatenate((predictBase,p_test_lgb), axis=0)
result = lgbResult.reshape(2,2556790).T
np.savetxt('LightGBMResult_submission_nofold.csv', result, delimiter=",",fmt=['%d','%0.8f'], header="id,target", comments="")

xgnResult = np.concatenate((predictBase,p_test_xgb), axis=0)
result2 = xgnResult.reshape(2,2556790).T
np.savetxt('XGBResult_submission_nofold.csv', result2, delimiter=",",fmt=['%d','%0.8f'], header="id,target", comments="")

catResult = np.concatenate((predictBase,p_test_cat), axis=0)
result3 = catResult.reshape(2,2556790).T
np.savetxt('CatBoostResult_submission_nofold.csv', result, delimiter=",",fmt=['%d','%0.8f'], header="id,target", comments="")

inverse_predict = np.concatenate((predictBase,predict_final[:,0]), axis=0)
result4 =  np.concatenate((predictBase,predict_final[:,1]), axis=0).reshape(2,2556790).T
np.savetxt("StackingResult.csv",result4, delimiter=",",fmt=['%d','%0.8f'], header="id,target", comments="")
