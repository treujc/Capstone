import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix

#file locations
current_dir = os.path.realpath(".")
data_folder = os.path.join(current_dir,'data')
datafile = os.path.join(data_folder,'CapstoneData.csv')
dataheaderfile = os.path.join(data_folder,'CapstoneDataHeaders.csv')
#Read in data files.
df_headers = list(pd.read_csv(dataheaderfile))
df = pd.read_csv(datafile,names = df_headers, low_memory = False)
#Trim data table
useful_columns = ['BusinessUnit','eventOccurredDatelday','event','eventType']
df_trimmed = df[useful_columns].reindex(columns = useful_columns)
#Clean data table
df_trimmed = df_trimmed[df_trimmed.BusinessUnit.isnull() == False]
df_trimmed.event.unique()
##categories can be considered counts over time. Which could then become ratios.
categories = ['event']
dfBUMatrix = pd.get_dummies(df_trimmed,columns = categories)
##Remove Central Support due to only having a few entries.
dfBUMatrix = dfBUMatrix[dfBUMatrix['BusinessUnit'] != 'Central Support']
##Remove Material Releases.
dfBUMatrix = dfBUMatrix[~dfBUMatrix['eventType'].str.contains('Material Release')]
dfBUMatrix['eventOccurredDatelday'] = pd.to_datetime(df['eventOccurredDatelday']).dt.date
dfBUMatrix['eventOccurredDatelday'] = pd.to_datetime(dfBUMatrix['eventOccurredDatelday'])
dfBUMatrix = dfBUMatrix.groupby([dfBUMatrix.eventOccurredDatelday,dfBUMatrix.BusinessUnit]).sum()
dfBUMatrix = dfBUMatrix.reset_index()
dfBUMatrix.sort_values(by = ['eventOccurredDatelday'])
dfFinal = dfBUMatrix.copy()
dfFinal = dfFinal.fillna(value = 0)
#dfFinal = dfFinal[(dfFinal['event_Observation'] != 0)]
dfFinal.rename(columns = {'eventOccurredDatelday':'DateKey'}, inplace = True)
dfFinal['WeekDay'] = dfFinal['DateKey'].dt.dayofweek
dfFinal.reset_index()


def add_rolling_sums(dfBU,DaysToRollList,DaysToPredict):
    '''
    Creates rolling sum columns based on a list of "Days Back" to roll the sums. 
    
    Input: A list of integers for witch you wish to calculate rolling sums for.
    
    Output: Adds columns to dataframe.
    '''
    for i in DaysToRollList:
        DaysToRoll = i
        event_Observation_Rolling = 'event_Observation_Rolling' + str(DaysToRoll)
        event_Incident_Rolling = 'event_Incident_Rolling' + str(DaysToRoll)
        dfBU[event_Observation_Rolling] = dfBU[['event_Observation']].groupby(dfBU['DateKey'] & dfBU['BusinessUnit']).apply(lambda g: g.rolling(DaysToRoll).sum().shift(1))
        dfBU[event_Incident_Rolling] = dfBU[['event_Incident']].groupby(dfBU['DateKey'] & dfBU['BusinessUnit']).apply(lambda g: g.rolling(DaysToRoll).sum().shift(1))
        dfBU['event_Incident_Prediction_Rolling'] = dfBU[['event_Incident']].groupby(dfBU['DateKey'] & dfBU['BusinessUnit']).apply(lambda g: g.rolling(DaysToPredict).sum().shift(-DaysToPredict))
    dfBU.reset_index()
    

if __name__ == '__main__':  
    DaysToRollList = [3,7,14,30,45]
    DaysToPredict = 14
    figure_size_l = 8.5
    figure_size_w = figure_size_l/.77
    ###Identify Predictors and what I am predicting.
    Predictors = [
                    'WeekDay',
                    'event_Incident_Rolling3',
                    'event_Incident_Rolling7',
                    'event_Incident_Rolling14',
                    'event_Incident_Rolling30',
                    'event_Incident_Rolling45',
                    'event_Observation_Rolling3',
                    'event_Observation_Rolling7',
                    'event_Observation_Rolling14',
                    'event_Observation_Rolling30',
                    'event_Observation_Rolling45'
                    ]
    To_Predict = ['event_Incident_Prediction_Rolling']
    seed = 9
    kfold = KFold(n_splits=3, random_state=seed, shuffle = True)
    for BU in ['East']:
        print('-------------------------------{} BU-----------------------------------'.format(BU))
        dfBU = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)].copy())
        add_rolling_sums(dfBU,DaysToRollList,DaysToPredict) 
        MinDate = '1/1/2018'
        dfBU = dfBU[lambda d: d.DateKey >= MinDate]   
        dfBUModel = dfBU[lambda d: pd.notnull(d.event_Incident_Prediction_Rolling) == True]
        MaxDate = '6/10/2018'
        dfBUModel = dfBUModel[lambda d: d.DateKey <= MaxDate] 
        X = pd.DataFrame(dfBUModel, columns=Predictors)
        y = np.ravel(pd.DataFrame(dfBUModel, columns=To_Predict))
        model = RandomForestRegressor(n_estimators = 40, random_state = seed)#,bootstrap = True)
        #Do initial Test-Train split.
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .2,random_state = seed, shuffle = True)
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)

#        X = pd.DataFrame(dfBU, columns=Predictors)
        predicted = pd.Series(model.predict(X))
        results = pd.DataFrame(dfBUModel, columns=['DateKey'])
        results['event_Incident_Prediction_Rolling'] = predicted.values
        results['MaxDate'] = MaxDate
        results.to_csv('results.csv', mode='a', header=False)
#        print(results)




#We have excluded "Material Release" from the dataset. 
#Consider Poisson Regression. #Seems very difficult to implement. Add this as a how to possibly improve note.