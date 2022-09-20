import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score
import xgboost
from sklearn.model_selection import train_test_split
import myMatrix as mM
import time
from numpy import genfromtxt
# -------------------------------------------------
# Nhat-Duc Hoang
# -------------------------------------------------
def CrossValidation(DataLoc = 'D:/Dataset1.csv', featureNum = 11, dataname = 'Dataset1'):
    # Load Data -------------------------------------------------
    # number of feature
    start = time.time()
    dataset	= genfromtxt(DataLoc, delimiter=',')
    X0	= dataset[:,0:featureNum]
    Y0	= dataset[:,featureNum]

    Nd = len(Y0)
    ridx = np.random.permutation(Nd)
    X0 = X0[ridx, :]
    Y0 = Y0[ridx]
    
    X, meanX, stdX = ZScoreNorm(X0)
    Y, meanY, stdY = ZScoreNormY(Y0)

    from sklearn.model_selection import KFold
    foldNum = 5;
    kf = KFold(n_splits=foldNum)
    kf.get_n_splits(X)

    max_depth_set = np.array([1, 2, 3, 4, 5])
    learning_rate_set = np.array([0.1, 0.3, 0.5, 1])
    reg_lambda_set = np.array([0.1, 1, 5, 10, 50, 100, 300])

    BestRmse = 100000;
    Best_max_depth = max_depth_set[0]
    Best_learning_rate = learning_rate_set[0]
    Best_reg_lambda = reg_lambda_set[0]

    for max_depth_ in max_depth_set:
        for learning_rate_ in learning_rate_set:      
            for Best_reg_lambda_ in reg_lambda_set:
                Sum_rmse_test = 0
                for train_index, test_index in kf.split(X):  
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    # Model Training ------------------------------------------------- 
                    PredictionModel = xgboost.XGBRegressor(learning_rate = learning_rate_, 
                                                           max_depth = max_depth_,
                                                           reg_lambda = Best_reg_lambda_)
                                   
                    # Model Prediction -------------------------------------------------
                    PredictionModel.fit(X_train, Y_train)
                    Y_test_pred = PredictionModel.predict(X_test)    
                    Y_test_pred = Y_test_pred*stdY + meanY  
                    Y_test = Y_test*stdY + meanY
                    rmse_test = mean_squared_error(Y_test, Y_test_pred, squared = False)
                    Sum_rmse_test = Sum_rmse_test + rmse_test           

                Average_rmse_test = Sum_rmse_test/foldNum
                print('Average_rmse_test');
                print(Average_rmse_test);
                if Average_rmse_test < BestRmse:
                    BestRmse = Average_rmse_test           
                    Bestmax_depth = max_depth_
                    Bestlearning_rate = learning_rate_ 
                    Best_reg_lambda = Best_reg_lambda_
        
    print('BestRmse');
    print(BestRmse);
    print('Bestmax_depth');
    print(Bestmax_depth);
    print('Bestlearning_rate');
    print(Bestlearning_rate);
    print('Best_reg_lambda');
    print(Best_reg_lambda);

    finishTime = time.time()
    ComputeCost = finishTime - start

    CV_Result = np.array([BestRmse, Bestmax_depth, Bestlearning_rate, Best_reg_lambda, ComputeCost])
    CV_Result_df = pd.DataFrame(CV_Result)
    CV_Result_df.to_csv('D:/XGBoostReg_CV_Result_' + dataname + '.csv')
    return Bestmax_depth, Bestlearning_rate, Best_reg_lambda
# -------------------------------------------------
def MultiRun_Regression_XGBoost(NR = 2, TestR = 0.1, lr = 0.3, MaxD = 3, RegLamda = 1,
                                DataLoc = 'D:/Dataset1.csv',
                                ExperimentName = 'MR_XGBoostReg_Dataset1',  featureNum = 11):
    start = time.time()
    
    mM.CreateFolder("D:/"+ ExperimentName)   
    CategorialVar = np.zeros(featureNum)   
    dataset	= genfromtxt(DataLoc, delimiter=',')
    X0	= dataset[:,0:featureNum]
    Y0	= dataset[:,featureNum]
 
    Nd = len(Y0)
    ridx = np.random.permutation(Nd)
    X0 = X0[ridx, :]
    Y0 = Y0[ridx]
    X, meanX, stdX = ZScoreNorm(X0)  
    Y, meanY, stdY = ZScoreNormY(Y0)
    
    RecordTrainingTime = np.zeros(NR)
    RecordPerformance = np.zeros((8, NR))

    Record_Y_train = np.zeros((1,1))
    Record_Y_test = np.zeros((1,1))
    Record_Y_train_p = np.zeros((1,1))
    Record_Y_test_p  = np.zeros((1,1))
    
    for r in range(NR):
        # np.random.seed(r)          
                    
        print("Current run = ", r)
        x_train, x_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size = TestR)   
        start_time = time.time()

        PredictionModel = xgboost.XGBRegressor(learning_rate = lr,\
                                           max_depth = MaxD,\
                                           reg_lambda = RegLamda)
       
        PredictionModel.fit(x_train, Y_train)
        end_time = time.time()
        RecordTrainingTime[r] = end_time-start_time     
        
        
        Y_train_p = PredictionModel.predict(x_train)
        Y_test_p = PredictionModel.predict(x_test)       

        Y_train_p = Y_train_p*stdY + meanY  
        Y_train = Y_train*stdY + meanY   
        
        Y_test_p = Y_test_p*stdY + meanY  
        Y_test = Y_test*stdY + meanY

        Ntr = Y_train.shape[0]
        Nte = Y_test.shape[0]
        Record_Y_train = np.vstack((Record_Y_train, Y_train.reshape(Ntr, 1)))
        Record_Y_test = np.vstack((Record_Y_test, Y_test.reshape(Nte, 1)))

        Record_Y_train_p = np.vstack((Record_Y_train_p, Y_train_p.reshape(Ntr, 1)))
        Record_Y_test_p = np.vstack((Record_Y_test_p, Y_test_p.reshape(Nte, 1)))
       
        RMSE_train = ComputeRmse(Y_train, Y_train_p)
        MAPE_train = ComputeMAPE(Y_train, Y_train_p)
        MAE_train = mean_absolute_error(Y_train, Y_train_p)
        R2_train = r2_score(Y_train, Y_train_p)          

        RMSE_test = ComputeRmse(Y_test, Y_test_p)
        MAPE_test = ComputeMAPE(Y_test, Y_test_p)
        MAE_test = mean_absolute_error(Y_test, Y_test_p)
        R2_test = r2_score(Y_test, Y_test_p)   
        
        RecordPerformance[:, r] = np.array([RMSE_train, MAPE_train, MAE_train,
                                            R2_train,
                                            RMSE_test, MAPE_test, MAE_test,
                                            R2_test])

    #print("RecordPerformance", RecordPerformance)

    mM.WriteCsv(RecordPerformance, "D:/" + ExperimentName + "/RecordPerformance.csv")    
    mM.WriteCsv(RecordTrainingTime, "D:/" + ExperimentName + "/RecordTrainingTime.csv")     

    Mean_RecordPerformance_Model = np.mean(RecordPerformance, axis = 1)
    mM.WriteCsv(Mean_RecordPerformance_Model, "D:/" + ExperimentName +"/Mean_RecordPerformance_Model.csv")
    Std_RecordPerformance_Model = np.std(RecordPerformance, axis = 1)
    mM.WriteCsv(Std_RecordPerformance_Model, "D:/" + ExperimentName +"/Std_RecordPerformance_Model.csv")
    print("Mean_RecordPerformance_Model = ", Mean_RecordPerformance_Model)

    Record_Y_train = np.delete(Record_Y_train, 0, axis = 0)
    Record_Y_test = np.delete(Record_Y_test, 0, axis = 0)

    Record_Y_train_p = np.delete(Record_Y_train_p, 0, axis = 0)
    Record_Y_test_p = np.delete(Record_Y_test_p, 0, axis = 0)

    mM.WriteCsv(Record_Y_train, "D:/" + ExperimentName +"/Record_Y_train.csv")
    mM.WriteCsv(Record_Y_test, "D:/" + ExperimentName +"/Record_Y_test.csv")
    mM.WriteCsv(Record_Y_train_p, "D:/" + ExperimentName +"/Record_Y_train_p.csv")
    mM.WriteCsv(Record_Y_test_p, "D:/" + ExperimentName +"/Record_Y_test_p.csv")

    finishTime = time.time()
    ComputeCost = finishTime - start
    print('ComputeCost = ', ComputeCost)
    print('AverageComputeCost = ', ComputeCost/NR)
    ComputeCostMat = np.array([ComputeCost])
    mM.WriteCsv(ComputeCostMat, "D:/" + ExperimentName +"/ComputeCostMat.csv")
    
    # return RecordPerformance, RecordTrainingTime
# -------------------------------------------------
def ComputeMse(Y, Yp):
    f = np.sum((Y-Yp)**2, 0)/Y.shape[0]
    return f
# ----------------------------------------------------
def ComputeRmse(Y, Yp):
    msef = np.sum((Y-Yp)**2, 0)/Y.shape[0]
    f = np.sqrt(msef)
    return f
# ----------------------------------------------------
def ComputeMAPE(Y, Yp):
    f = 100*np.sum(np.abs((Y-Yp)/Y), 0)/Y.shape[0]
    return f
# ----------------------------------------------------
def ZScoreNorm(X = np.random.random((20, 2))*10):
    # Z score normalization
    MeanX = np.mean(X, axis = 0)
    StdX  = np.std(X, axis = 0)
    nX = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            nX[i,j] = (X[i,j] - MeanX[j])/StdX[j]
    return nX, MeanX, StdX
def ZScoreNormY(X = np.random.random(20)):
    # Z score normalization
    MeanX = np.mean(X)
    StdX  = np.std(X)
    nX = np.zeros(X.shape[0])
    for i in range(X.shape[0]):        
        nX[i] = (X[i] - MeanX)/StdX
    return nX, MeanX, StdX
# -------------------------------------------------
featureNum_ = 11
DataLoc_ = 'D:/Dataset1.csv'
dataname_ = 'Dataset1'

Bestmax_depth, Bestlearning_rate, Best_reg_lambda = \
CrossValidation(DataLoc = DataLoc_, featureNum = featureNum_, dataname = dataname_)

MultiRun_Regression_XGBoost(NR = 20, TestR = 0.15, lr = Bestlearning_rate, MaxD = Bestmax_depth,
                            RegLamda = Best_reg_lambda,
                            DataLoc = DataLoc_,
                            ExperimentName = dataname_,
                            featureNum = featureNum_)
# -------------------------------------------------






















