import pandas as pd
import numpy as np
import os
import MyFunctions
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
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
#Create dfDates dataframe based on max and min values in main dataframe.
end_date = dfBUMatrix['eventOccurredDate'].max()
start_date = dfBUMatrix['eventOccurredDate'].min()
dfDates = MyFunctions.create_dates_dataframe(start_date,end_date)
#Grab the distinct list of BU's.
dfBUList = pd.DataFrame(dfBUMatrix.BusinessUnit.unique(),columns = ['BusinessUnit'])
#Spread the dates across all BU's.
dfCounts = MyFunctions.dataframe_crossjoin(dfDates, dfBUList)
#Spread the counts across all dates even when you have zero for the count.
dfFinal = pd.merge(dfCounts, dfBUMatrix, left_on=['DateKey','BusinessUnit'],right_on=['eventOccurredDate','BusinessUnit'], how='left')
dfFinal = dfFinal.fillna(value = 0)
dfFinal.drop('eventOccurredDate',axis = 1,inplace = True)


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
        dfFinal[event_Observation_Rolling] = dfFinal[['event_Observation']].groupby(dfFinal['DateKey'] & dfFinal['BusinessUnit']).apply(lambda g: g.rolling(DaysToRoll).sum().shift(1))
        dfFinal[event_Incident_Rolling] = dfFinal[['event_Incident']].groupby(dfFinal['DateKey'] & dfFinal['BusinessUnit']).apply(lambda g: g.rolling(DaysToRoll).sum().shift(1))
    dfFinal.reset_index()


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

def run_model(model):
    '''
    Runs the model passed into the function through each BU.
    
    Input: model
    
    Output: r2.
    '''
    for BU in list(dfFinal.BusinessUnit.unique()):#list(dfFinal.BusinessUnit.unique()):
        X = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)], columns=Predictors)
        y = np.ravel(pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)], columns=To_Predict))
        results = cross_val_score(model, X, y, cv=kfold)
        predicted = cross_val_predict(model, X, y, cv=kfold)
        r2 = results.mean()
        plt.figure(figsize= (10,10))
        ax = plt.subplot()
        ax.scatter(y, predicted, edgecolors=(.5,1,0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        plt.title('{} BU, r2: {:.2f}'.format(BU,r2))
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        #print('{} BU, r2: {:.2f}'.format(BU,r2))
        plt.show()


if __name__ == '__main__':   
    DaysToRollList = [7]
    add_rolling_sums(DaysToRollList)
    MinDate = '12/1/2017'
    dfFinal = dfFinal[(dfFinal['DateKey'] >= MinDate)]
    dfFinal.reset_index()
    ###Identify Predictors and what I am predicting.
    Predictors = ['event_Observation_Rolling7']
    To_Predict = ['event_Incident_Rolling7']
    seed = 9
    kfold = KFold(n_splits=3, random_state=seed, shuffle = True)
    print('--------------------Linear Regression----------------------')
    model = LinearRegression()
    run_model(model)
    print('-----------------------------------------------------------')
    print('------------------Random Forest Regressor------------------')
    model = RandomForestRegressor(n_estimators = 50)#,bootstrap = True)
    run_model(model)
    print('-----------------------------------------------------------')
    print('--------------------Ridge Regression--------------------')
    model = Ridge(alpha = .5,normalize = True)
    run_model(model)
    print('-----------------------------------------------------------')
    
#    #Create a Scatter Matrix.
#    print(scatter_matrix(dfFinal, figsize = (12,12))) 
    
    
    


