import pandas as pd
import numpy as np
import os
import MyFunctions
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
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
###This line shows the behavior we hoped to see.
##dfFinal['event_Incident'] = -dfFinal['event_Incident']



def add_rolling_sums(DaysToRollList,DaysToPredict):
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
        dfFinal['event_Incident_Prediction_Rolling'] = dfFinal[['event_Incident']].groupby(dfFinal['DateKey'] & dfFinal['BusinessUnit']).apply(lambda g: g.rolling(DaysToPredict).sum().shift(-DaysToPredict))
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

def run_model(model,dfBU):
    '''
    Runs the model passed into the function.
    
    Input:  model = Type of model.
            ,dfBU = A dataframe trimmed down to one BU.
    
    Output: 
    '''
    X = pd.DataFrame(dfBU, columns=Predictors)
    y = np.ravel(pd.DataFrame(dfBU, columns=To_Predict))
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
    plt.suptitle('Measured vs Predicted for {} BU'.format(BU),fontsize = 12)
    plt.title('r2: {:.2f}'.format(r2),fontsize = 12)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    #print('{} BU, r2: {:.2f}'.format(BU,r2))
    plt.show()
    #Plot over time where actuals are a scatter plot and prediction is the line.
    #plt.figure(figsize= (figure_size,figure_size))
    #plt.plot(predicted)



if __name__ == '__main__':  
    DaysToRollList = [3,7,15,30]
    DaysToPredict = 7
    add_rolling_sums(DaysToRollList,DaysToPredict)
    MinDate = '12/1/2017'
    dfFinal = dfFinal[(dfFinal['DateKey'] >= MinDate)]
    dfFinal = dfFinal[(dfFinal['event_Observation'] != 0)]
    dfFinal = dfFinal[dfFinal.event_Incident_Prediction_Rolling.isnull() == False]
    dfFinal['WeekDay'] = dfFinal['DateKey'].dt.dayofweek
    dfFinal.reset_index()
    figure_size = 5
    ###Identify Predictors and what I am predicting.
    Predictors = [#'WeekDay',
                  #'event_Incident_Rolling3',
                  'event_Incident_Rolling7',
                    'event_Incident_Rolling15',
                  'event_Incident_Rolling30',
                  # 'event_Observation_Rolling3',
                    'event_Observation_Rolling7',
                    'event_Observation_Rolling15',
                    'event_Observation_Rolling30']
    To_Predict = ['event_Incident_Prediction_Rolling']
    seed = 9
    kfold = KFold(n_splits=3, random_state=seed, shuffle = True)
    for BU in list(dfFinal.BusinessUnit.unique()):  #['Midcon']:
        dfBU = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)])
#        #Scatter matrix of correlations.
#        scatter_matrix(dfBU, alpha=0.4, figsize=(figure_size,figure_size), diagonal='kde')
#        plt.suptitle('Scatter Matrix for {} BU'.format(BU),fontsize = 12)
#        plt.show()
        print('-------------------------Linear Regression---------------------------')
        model = LinearRegression()
        run_model(model,dfBU)
        print('---------------------------------------------------------------------')
#        print('-----------------------Random Forest Regressor-----------------------')
#        model = RandomForestRegressor(n_estimators = 50)#,bootstrap = True)
#        run_model(model,dfBU)
#        print('---------------------------------------------------------------------')
#        print('---------------------------Ridge Regression--------------------------')
#        model = Ridge(alpha = .5,normalize = True)
#        run_model(model,dfBU)
#        print('---------------------------------------------------------------------')

    
    
    


