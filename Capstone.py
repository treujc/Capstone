import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#file locations
current_dir = os.path.realpath(".")
data_folder = os.path.join(current_dir,'data')
datafile = os.path.join(data_folder,'CapstoneData.csv')
dataheaderfile = os.path.join(data_folder,'CapstoneDataHeaders.csv')
#Read in data files.
df_headers = list(pd.read_csv(dataheaderfile))
df = pd.read_csv(datafile,names = df_headers, low_memory = False)
#Trim data table
useful_columns = ['BusinessUnit','eventOccurredDate','event']
df_trimmed = df[useful_columns].reindex(columns = useful_columns)
#Clean data table
df_trimmed = df_trimmed[df_trimmed.BusinessUnit.isnull() == False]
df_trimmed.event.unique()
##categories can be considered counts over time. Which could then become ratios.
categories = ['event']
dfBUMatrix = pd.get_dummies(df_trimmed,columns = categories)
##Remove Central Support due to only having a few entries.
dfBUMatrix = dfBUMatrix[dfBUMatrix['BusinessUnit'] != 'Central Support']
dfBUMatrix['eventOccurredDate'] = pd.to_datetime(df['eventOccurredDate']).dt.date
dfBUMatrix['eventOccurredDate'] = pd.to_datetime(dfBUMatrix['eventOccurredDate'])
dfBUMatrix = dfBUMatrix.groupby([dfBUMatrix.eventOccurredDate,dfBUMatrix.BusinessUnit]).sum()
dfBUMatrix = dfBUMatrix.reset_index()
dfBUMatrix.sort_values(by = ['eventOccurredDate'])
dfFinal = dfBUMatrix.copy()
dfFinal = dfFinal.fillna(value = 0)
dfFinal = dfFinal[(dfFinal['event_Observation'] != 0)]
dfFinal.rename(columns = {'eventOccurredDate':'DateKey'}, inplace = True)
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
    

#def rmse(true, predicted):
#    '''
#    Calculates Root Mean Square Error.
#    
#    Input: true = Observed values, predicted = Results from model prediction.
#    
#    Output: RMSE.
#    '''
#    diff = (true - predicted)
#    sumsq = np.sum(diff**2)
#    count = diff.shape[0]
#    result = np.sqrt(sumsq/count)
#    return result



if __name__ == '__main__':  
    DaysToRollList = [3,7,14,30,45]
    DaysToPredict = 14
    figure_size = 5
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
    for BU in list(dfFinal.BusinessUnit.unique()):  #['Midcon']:
        print('-------------------------------{} BU-----------------------------------'.format(BU))
        dfBU = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)].copy())
        add_rolling_sums(dfBU,DaysToRollList,DaysToPredict) 
        MinDate = '1/1/2018'
        dfBU = dfBU[lambda d: d.DateKey >= MinDate]   
        dfBU = dfBU[lambda d: pd.notnull(d.event_Incident_Prediction_Rolling) == True]
        #Scatter matrix of correlations.
#        scatter_matrix(dfBU, alpha=0.4, figsize=(figure_size,figure_size), diagonal='kde')
#        plt.suptitle('Scatter Matrix for {} BU'.format(BU),fontsize = 12)
#        plt.show()
        X = pd.DataFrame(dfBU, columns=Predictors)
        y = np.ravel(pd.DataFrame(dfBU, columns=To_Predict))
        model = RandomForestRegressor(n_estimators = 50)#,bootstrap = True)
        #Do initial Test-Train split.
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .2,random_state = seed)
        results = cross_val_score(model, X_train, y_train, cv=kfold)
        predicted = cross_val_predict(model, X_train, y_train, cv=kfold)
        r2 = results.mean()
        #Scatter plot of Predicted vs Measured.
        plt.figure(figsize= (figure_size,figure_size))
        ax = plt.subplot()
        ax.scatter(y_train, predicted, edgecolors=(.5,1,0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        plt.suptitle('Measured vs Predicted, r2: {:.2f}'.format(r2),fontsize = 12)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        #Plot Feature Importances
        model.fit(X_train,y_train)
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='r', align='center')
        plt.yticks(range(len(indices)), Predictors)
        plt.xlabel('Relative Importance')
        plt.show()
        # The baseline predictions are the historical averages
        baseline_preds = X_test['event_Incident_Rolling14']
        # Baseline errors, and display average baseline error
        baseline_errors = abs(baseline_preds - y_test)
        print('Average baseline error: ', round(np.mean(baseline_errors), 2))
        # Use the forest's predict method on the test data
        predictions = model.predict(X_test)
        # Calculate the absolute errors
        errors = abs(predictions - y_test)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#Are there different types of Observations? Maybe bucket Spills differently than cuts etc. 
#Consider Poisson Regression. #Seems very difficult to implement. Add this as a how to possibly improve note.
        
        
    
    


