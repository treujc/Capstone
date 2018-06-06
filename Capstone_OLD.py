import pandas as pd
import numpy as np
import os
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder, Binarizer
import MyFunctions
from sklearn.model_selection import train_test_split
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def load_data():
    '''
    Loads in the data from file.
    '''
    #file locations
    current_dir = os.path.realpath(".")
    data_folder = os.path.join(current_dir,'data')
    datafile = os.path.join(data_folder,'CapstoneData.csv')
    dataheaderfile = os.path.join(data_folder,'CapstoneDataHeaders.csv')
    #Read in data files.
    df_headers = list(pd.read_csv(dataheaderfile))
    df = pd.read_csv(datafile,names = df_headers, low_memory = False)
    return df

def clean_data():
    '''
    Cleans the data frame to prepare it for analysis.
    '''
#Trim data table
    useful_columns = ['BusinessUnit','eventOccurredDate','companyInvolved','operationOrDevelopment'
                          ,'jobTypeObserved','event','eventClassification','eventType','stopJob'
                          ,'immediateActionsTaken','actionCompletedOnsiteDetail','furtherActionNecessaryComments','rigInvolved']
    df_trimmed = df.copy()
    df_trimmed = df_trimmed[useful_columns]
    #Clean data table
    df_trimmed[['rigInvolved']] = df_trimmed[['rigInvolved']].fillna(value='No')
    df_trimmed[['stopJob']] = df_trimmed[['stopJob']].fillna(value='No')
    df_trimmed.loc[(df_trimmed['immediateActionsTaken'] == 'Action Completed Onsite'),'furtherActionNecessaryComments'] = np.nan
    df_trimmed.loc[(df_trimmed['immediateActionsTaken'] == 'Further Action Necessary'),'actionCompletedOnsiteDetail'] = np.nan
    df_trimmed['comments'] = df_trimmed.actionCompletedOnsiteDetail.combine_first(df_trimmed.furtherActionNecessaryComments)
    df_trimmed[['comments']] = df_trimmed[['comments']].fillna(value='None')
    df_trimmed = df_trimmed.drop(['actionCompletedOnsiteDetail', 'furtherActionNecessaryComments'], axis=1)
    df_trimmed = df_trimmed[df_trimmed.companyInvolved.isnull() == False]
    df_trimmed = df_trimmed[df_trimmed.jobTypeObserved.isnull() == False]
    df_trimmed = df_trimmed[df_trimmed.immediateActionsTaken.isnull() == False]
    df_trimmed = df_trimmed[df_trimmed.BusinessUnit.isnull() == False]
    df_trimmed.loc[(df_trimmed['companyInvolved'] != 'BP'),'companyInvolved'] = 'Other'
    df_trimmed.event.unique()
    ##categories can be considered counts over time. Which could then become ratios.
    categories = ['event','eventClassification']#,'companyInvolved','operationOrDevelopment','jobTypeObserved','stopJob','immediateActionsTaken','rigInvolved']
    dfBUMatrix = pd.get_dummies(df_trimmed,columns = categories)
    useful_columns = ['BusinessUnit','eventOccurredDate','event_Observation','event_Incident']
    dfBUMatrix = dfBUMatrix[useful_columns]
    dfBUMatrix = dfBUMatrix[dfBUMatrix['BusinessUnit'] != 'Central Support']
    dfBUMatrix['eventOccurredDate'] = pd.to_datetime(df['eventOccurredDate']).dt.date
    dfBUMatrix['eventOccurredDate'] = pd.to_datetime(dfBUMatrix['eventOccurredDate'])
    dfBUMatrix = dfBUMatrix.groupby([dfBUMatrix.eventOccurredDate,dfBUMatrix.BusinessUnit]).sum()
    dfBUMatrix = dfBUMatrix.reset_index()
    dfBUMatrix.sort_values(by = ['eventOccurredDate'])
    BUList = dfBUMatrix.BusinessUnit.unique()
    #Remove Central Support due to only having a few entries.
    BUList = BUList[BUList != 'Central Support']
    #Create dfDates dataframe based on max and min values in main dataframe.
    end_date = dfBUMatrix['eventOccurredDate'].max()
    start_date = dfBUMatrix['eventOccurredDate'].min()
    dfDates = MyFunctions.create_dates_dataframe(start_date,end_date)
    #Grab the distinct list of BU's.
    dfBUList = pd.DataFrame(BUList,columns = ['BusinessUnit'])
    #Spread the dates across all BU's.
    dfCounts = MyFunctions.dataframe_crossjoin(dfDates, dfBUList)
    #Spread the counts across all dates even when you have zero for the count.
    dfFinal = pd.merge(dfCounts, dfBUMatrix, left_on=['DateKey','BusinessUnit'],right_on=['eventOccurredDate','BusinessUnit'], how='left')
    dfFinal = dfFinal.fillna(value = 0)
    dfFinal.drop('eventOccurredDate',axis = 1,inplace = True)
    return dfFinal


def add_rolling_sums(DaysToRollList):
    '''
    Creates rolling sum columns based on a list of "Days Back" to roll the sums. 
    
    Input: A list of integers for witch you wish to calculate rolling sums for.
    
    Output: Adds columns to dataframe.
    '''
    for i in DaysToRollList:
        DaysToRoll = i
        event_Observation_Rolling = 'event_Observation_Rolling' + str(DaysToRoll)
        event_Incident_Rolling = 'event_Incident_Rolling' + str(DaysToRoll)
        dfFinal[event_Observation_Rolling] = dfFinal[['event_Observation']].groupby(dfFinal['DateKey'] & dfFinal['BusinessUnit']).apply(lambda g: g.rolling(DaysToRoll).sum().shift(7))
        dfFinal[event_Incident_Rolling] = dfFinal[['event_Incident']].groupby(dfFinal['DateKey'] & dfFinal['BusinessUnit']).apply(lambda g: g.rolling(DaysToRoll).sum().shift(7))


def rmse(true, predicted):
    '''
    Calculates Root Mean Square Error.
    
    Input: true = Observed values, predicted = Results from model prediction.
    
    Output: RMSE.
    '''
    diff = (true - predicted)
    sumsq = np.sum(diff**2)
    count = diff.shape[0]
    result = np.sqrt(sumsq/count)
    return result

def linear_regression():
    for BU in BUList:
        #Create dataframes to represent what I am using as a predictor X vs what I hope to predict y.
        X = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)], columns=Predictors)
        y = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)], columns=To_Predict)
#        #Perform Test-Train split on a 20% break.
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
#        # Fit your model using the training set
#        model = LinearRegression()
#        results = model.fit(X_train, y_train)
#        # Call predict to get the predicted values for training and test set
#        test_predicted = results.predict(X_test)
#        # Calculate RMSE for the test set
#        RMSE = float(rmse(y_test, test_predicted))
#        r2 = r2_score(y_test,test_predicted)
#        print('{} BU, RMSE: {:.2f}, r2: {:.2f}'.format(BU,float(RMSE),r2))
        results = cross_val_score(model, X, y, cv=kfold)
        print(results.mean())


##Make sure to throw out current day. See if there is kwarg to do this. ##.shift(1) accomplished this.
##Try regularized regressor and random forrest regressor. Linear Regression.
##Need to store results. Need to keep track of what works best.



if __name__ == '__main__':
    df = load_data()
    dfFinal = clean_data()    
    DaysToRollList = [7,15]#,30,45,60,90]
    add_rolling_sums(DaysToRollList)
    #Remove columns less than minimum bucket we rolled into. This trims off NAN values.
    #DaysToRemove = int(np.min(DaysToRollList))
    #start_date = dfFinal['DateKey'].min()
    #MinDate = start_date + timedelta(days=DaysToRemove)
    MinDate = '12/1/2017' ##Remove this line and use the above if you want to mythodically set the first date value.
    dfFinal = dfFinal[(dfFinal['DateKey'] >= MinDate)]
    #Identify Predictors and what I am predicting.
    Predictors = ['event_Observation','event_Observation_Rolling7']
    To_Predict = ['event_Incident_Rolling7']
    # create pipeline
    estimators = []
    #estimators.append(('Linear', LinearRegression()))
    estimators.append(('RandomForestRegressor', RandomForestRegressor()))
    model = Pipeline(estimators)
    # evaluate pipeline
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed)
    print('--------------------Linear Regression----------------------')
    linear_regression()
    print('-----------------------------------------------------------')
    print(estimators)


# =============================================================================
#  #Create a Scatter Matrix.
#  from pandas.plotting import scatter_matrix
#  print(scatter_matrix(dfFinal, figsize = (12,12)))
#  plt.show()
# =============================================================================




# =============================================================================
# def crossVal(X_train, y_train):
#     rmseList = []
#     kf = KFold(n_splits=5)
#     #kf.get_n_splits(X)
#     #print(kf)
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         y_pred = model.fit(X_train,y_train).predict(X_test)
#         rmseList.append(rmse(y_test,y_pred))
#         result = np.mean(rmseList)
#     return(result)
#     
# def my_cross_val_scores(X_data, y_data, num_folds=3):
#     ''' Returns error for k-fold cross validation. '''
#     kf = KFold(n_splits=num_folds)
#     train_error = np.empty(num_folds)
#     test_error = np.empty(num_folds)
#     index = 0
#     linear = LinearRegression()
#     for train, test in kf.split(X_data):
#         linear.fit(X_data[train], y_data[train])
#         pred_train = linear.predict(X_data[train])
#         pred_test = linear.predict(X_data[test])
#         train_error[index] = rmse(pred_train, y_data[train])
#         test_error[index] = rmse(pred_test, y_data[test])
#         index += 1
#     return np.mean(test_error), np.mean(train_error) 
# =============================================================================







