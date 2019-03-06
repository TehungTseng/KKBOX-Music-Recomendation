import pandas as pd
import numpy as np
import math
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split,KFold
from sklearn import metrics, ensemble,linear_model
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib as mpl
import xgboost
import lightgbm
import warnings
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

class ComprehensiveClassifier:
    def __init__(self, nfolds ,models = ['cb'], percent = 0.01):
        self.nfolds = nfolds
        self.models = models
        self.precent = percent

    def dataPreprocess(self):
        train = pd.read_csv("./data/train.csv")
        train = train.sample(frac=self.precent)
        self.testData = pd.read_csv("./data/test.csv")
        songs = pd.read_csv('./data/songs.csv')
        train = pd.merge(train, songs, on='song_id', how='left')
        self.testData = pd.merge(self.testData,songs,on='song_id',how='left')
        del songs
        
        self.predictBase = np.array(self.testData.pop('id'))

        members = pd.read_csv('./data/members.csv')
        train = pd.merge(train, members, on='msno', how='left')
        self.testData = pd.merge(self.testData, members, on='msno', how='left')
        del members

        train.isnull().sum()/train.isnull().count()*100
        self.testData.isnull().sum()/self.testData.isnull().count()*100

        for i in train.select_dtypes(include=['object']).columns:
            train[i][train[i].isnull()] = 'unknown'

        for i in self.testData.select_dtypes(include=['object']).columns:
            self.testData[i][self.testData[i].isnull()] = 'unknow'
        
        train = train.fillna(value=0)
        self.testData = self.testData.fillna(value=0)

        train.registration_init_time = pd.to_datetime(train.registration_init_time, format='%Y%m%d', errors='ignore')
        train['registration_init_time_year'] = train['registration_init_time'].dt.year
        train['registration_init_time_month'] = train['registration_init_time'].dt.month
        train['registration_init_time_day'] = train['registration_init_time'].dt.day
        train['registration_init_time'] = train['registration_init_time'].astype('category')

        self.testData.registration_init_time = pd.to_datetime(self.testData.registration_init_time, format='%Y%m%d', errors='ignore')
        self.testData['registration_init_time_year'] = self.testData['registration_init_time'].dt.year
        self.testData['registration_init_time_month'] = self.testData['registration_init_time'].dt.month
        self.testData['registration_init_time_day'] = self.testData['registration_init_time'].dt.day
        self.testData['registration_init_time'] = self.testData['registration_init_time'].astype('category')

        # Object data to category
        for col in train.select_dtypes(include=['object']).columns:
            train[col] = train[col].astype('category')
    
        # Encoding categorical features
        for col in train.select_dtypes(include=['category']).columns:
            train[col] = train[col].cat.codes
        
        # Object data to category
        for col in self.testData.select_dtypes(include=['object']).columns:
            self.testData[col] = self.testData[col].astype('category')
    
        # Encoding categorical features
        for col in self.testData.select_dtypes(include=['category']).columns:
            self.testData[col] = self.testData[col].cat.codes

        train = train.drop(['expiration_date', 'lyricist'], 1)

        self.testData = self.testData.drop(['expiration_date', 'lyricist'], 1)

        model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=25)
        model.fit(train[train.columns[train.columns != 'target']], train.target)

        df_plot = pd.DataFrame({'features': train.columns[train.columns != 'target'],
                        'importances': model.feature_importances_})
        df_plot = df_plot.sort_values('importances', ascending=False)

        train = train.drop(df_plot.features[df_plot.importances < 0.04].tolist(), 1)
        self.testData = self.testData.drop(df_plot.features[df_plot.importances < 0.04].tolist(), 1)

        target = train.pop('target')

        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(train, target, train_size=0.9, random_state=1234)
        
        del train

    def stacking_train_test_split (self,data, pm_data, iteration_time):

        np_x_data = np.array(data)
        np_y_data = np.array(pm_data)

        start_flag = int((iteration_time)*0.2*(len(np_x_data)-1))
        end_flag = int((iteration_time+1.0)*0.2*(len(np_x_data)-1))

        if (iteration_time ==0):
            X_train = np_x_data[end_flag+ 1: ,:]  
            y_train = np_y_data[end_flag+ 1: ]
            X_test = np_x_data[0: end_flag,:]

        elif (iteration_time == 4):
            X_train = np_x_data[0: start_flag-1,:]  
            y_train = np_y_data[0: start_flag-1]
            X_test = np_x_data[start_flag: ,:]
        else :
            X_train = np.concatenate((np_x_data[0: start_flag-1,:], np_x_data[end_flag+ 1: ,:]), axis=0) 
        
            y_train = np.concatenate((np_y_data[0: start_flag-1], np_y_data[end_flag+ 1:]), axis=0) 

            X_test = np_x_data[start_flag: end_flag, :]    

        return X_train, y_train, X_test

    def xgboost_for_stack (self,X_train, y_train, X_test):
        #250
        xgb = xgboost.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5, n_estimators=1,verbose_eval=True,silent=False)

        xgb.fit(X_train, y_train)
        predictions_xgb_test = xgb.predict_proba(X_test)[:,0]
        predictions_xgb_final = xgb.predict_proba(self.testData)[:,0]

        return predictions_xgb_test, predictions_xgb_final

    def lightgbm_for_stack (self, X_train, y_train, X_test):
        ###########################light gbm###########################
        params = {}
        params['learning_rate'] = 0.5
        params['application'] = 'binary'
        params['max_depth'] = 15
        params['num_leaves'] = 2**6
        params['verbosity'] = 0
        params['metric'] = 'auc'
        d_train = lightgbm.Dataset(X_train, y_train)
        watchlist = [d_train]
        lgb = lightgbm.train(params, train_set=d_train, num_boost_round=1, valid_sets=watchlist,verbose_eval=5)


        #lgb = lightgbm.LGBMClassifier(num_boost_round=60, learning_rate= 0.1, max_depth = 10, num_leaves = 2**6, verbosity = 0 ,metric= 'auc')
    
        #lgb.fit(X_train, y_train)
    
        
        predictions_lgb_test = lgb.predict_proba(X_test)[:,0]  
        predictions_lgb_final = lgb.predict_proba(self.testData)[:,0]
        ##################################################################
    
        return predictions_lgb_test, predictions_lgb_final

    def stacking_train(self,X_train_pre, y_train_pre, X_test_pre, model):
        if model == 'xgb':
            stacking_test, stacking_final = self.xgboost_for_stack(self.train_X, self.train_Y, self.test_X)
        else:
            stacking_test, stacking_final = self.lightgbm_for_stack(self.train_X, self.train_Y, self.test_X)

        name = "XGBoostResult.csv" if model == 'xgb' else "LightGBMResult.csv"
        predictionResult = np.concatenate((self.predictBase,stacking_final), axis=0)
        result = predictionResult.reshape(2,2556790).T
        np.savetxt(name, result, delimiter=",",fmt=['%d','%0.8f'], header="id,target", comments="")
        return  stacking_test, stacking_final


    def train(self):
        self.models_kf_return = {}

        # ------------------------------------------------------------------- Layer 1 -------------------------------------------------------------------
        kf_test = np.zeros(self.test_X.shape[0]) #生成测试集预测结果的容器
        kf_final_test = np.zeros(self.testData.shape[0]) #生成测试集预测结果的容器
        kf_test_predict = np.empty([self.test_X.shape[0]]) #生成测试集多折预测结果的容器
        kf_final_predict = np.empty([self.testData.shape[0]]) #生成测试集多折预测结果的容器
        
        clf = CatBoostClassifier(iterations=1, learning_rate=0.1, depth=10, loss_function='Logloss')

        clf.fit(self.train_X, self.train_Y) #折数训练

        kf_test_predict = clf.predict_proba(self.test_X)[:,0] #预测测试集
        kf_final_predict = clf.predict_proba(self.testData)[:,0]

        # ----------------------------------  CatBoost Region ----------------------------------

        predictionResult = np.concatenate((self.predictBase,kf_final_predict), axis=0)
        result = predictionResult.reshape(2,2556790).T
        np.savetxt("CatBoostResult.csv", result, delimiter=",",fmt=['%d','%0.5f'], header="id,target", comments="")

        cb_kf_test = kf_test_predict.reshape(-1, 1)
        cb_kf_final_test = kf_final_predict.reshape(-1,1)
        self.models_kf_return['cb'] = [ cb_kf_test,cb_kf_final_test]
        # ---------------------------------- CatBoost Region ----------------------------------

        stacked_y_test, stacked_final_test = self.stacking_train(self.train_X, self.train_Y, self.test_X,'xgb')
        self.models_kf_return['xgb'] = [stacked_y_test.reshape(-1,1),stacked_final_test.reshape(-1,1)]

        stacked_y_test_lgb, stacked_final_test = self.stacking_train(self.train_X, self.train_Y, self.test_X, 'gbm')
        self.models_kf_return['gbm'] = [stacked_y_test_lgb.reshape(-1,1),stacked_final_test.reshape(-1,1)]
        
        # ---------------------------------- Data Concatenate ---------------------------------- 
        for i in range(len(self.models)):
            if i == 0:
                self.final_test = self.models_kf_return[self.models[i]][0]
                self.final_predict = self.models_kf_return[self.models[i]][1]
            else:
                self.final_test = np.concatenate((self.final_test,self.models_kf_return[self.models[i]][0]), axis=1)
                self.final_predict = np.concatenate((self.final_predict,self.models_kf_return[self.models[i]][1]), axis=1)
        # ---------------------------------- Data Concatenate ---------------------------------- 
    def getSubmitBase(self):
        return self.predictBase

    def predict(self):

        '''
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(self.final_train.reshape(-1,3), np.array(self.train_Y).reshape(-1,1))

        # Make predictions using the testing set
        predictions_ligr = regr.predict(np.array(self.final_test).reshape(-1,3))
        #print(predictions_ligr)
        print("RMSE: %.2f" % math.sqrt(np.mean((predictions_ligr - np.array(self.test_Y).reshape(-1,1)) ** 2)))
        '''
        
        svm = SVC(kernel='linear',probability=True)
        '''
        svm.fit(self.final_train.reshape(-1,3),np.array(self.train_Y).reshape(-1,1))

        print(metrics.classification_report(np.array(self.test_Y).reshape(-1,1), svm.predict(np.array(self.final_test).reshape(-1,3))))
        '''
        logistic_regression = LogisticRegression()
        
        logistic_regression.fit(self.final_test.reshape(-1,len(self.models)),np.array(self.test_Y).reshape(-1,1))

        #predict_labels = logistic_regression.predict_proba(np.array(self.final_test).reshape(-1,len(self.models)))
        predict_final = logistic_regression.predict_proba(np.array(self.final_predict).reshape(-1,len(self.models)))

        self.predictBase = np.concatenate((self.predictBase,predict_final[:,0]), axis=0)
        #for i in range(len(predict_final)):
        #    predictionResult[i] = [i,predict_final[i][0]]
        result = self.predictBase.reshape(2,2556790).T
        np.savetxt("result.csv", result, delimiter=",",fmt=['%d','%0.8f'], header="id,target", comments="")
        #print(metrics.classification_report(np.array(self.test_Y).reshape(-1,1), predict_labels_after))


        '''
        vModel = VotingClassifier(estimators=[('svm',svm),('lr',logistic_regression)],voting='hard')
        vModel.fit(self.final_train.reshape(-1,len(self.models)),np.array(self.train_Y).reshape(-1,1))
        print(vModel.predict(np.array(self.final_predict).reshape(-1,len(self.models))))
        print(metrics.classification_report(np.array(self.test_Y).reshape(-1,1), vModel.predict(np.array(self.final_test).reshape(-1,len(self.models)))))
        '''

