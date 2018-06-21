import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 16.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})

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
#MaxDate = '5/30/2018'
#dfFinal = dfFinal[lambda d: d.DateKey < MaxDate] 
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
    for BU in list(dfFinal.BusinessUnit.unique()):  #['Midcon']:
        print('-------------------------------{} BU-----------------------------------'.format(BU))
        dfBU = pd.DataFrame(dfFinal[(dfFinal['BusinessUnit'] == BU)].copy())
        add_rolling_sums(dfBU,DaysToRollList,DaysToPredict) 
        MinDate = '1/1/2018'
        dfBU = dfBU[lambda d: d.DateKey >= MinDate]   
        dfBUModel = dfBU[lambda d: pd.notnull(d.event_Incident_Prediction_Rolling) == True]
#        #Scatter matrix of correlations.
#        scatter_matrix(dfBU, alpha=0.4, figsize=(30,30), diagonal='kde')
#        plt.suptitle('Scatter Matrix for {} BU'.format(BU),fontsize = 8)
#        plt.show()
        X = pd.DataFrame(dfBUModel, columns=Predictors)
        y = np.ravel(pd.DataFrame(dfBUModel, columns=To_Predict))
        model = RandomForestRegressor(n_estimators = 40, random_state = seed)#,bootstrap = True)
        #Do initial Test-Train split.
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .2,random_state = seed, shuffle = True)
        results = cross_val_score(model, X_train, y_train, cv=kfold)
        r2 = results.mean()
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        train_predicted = model.predict(X_train)
        # The baseline predictions are the historical averages
        baseline_preds = X_test['event_Incident_Rolling14']
        # Baseline errors, and display average baseline error
        baseline_errors = abs(baseline_preds - y_test)
#        print('Average baseline error: ', round(np.mean(baseline_errors), 2))
        # Calculate the absolute errors
        errors = abs(predicted - y_test)
#        print('Mean Absolute Error:', round(np.mean(errors), 2))
        #Scatter plot of Predicted vs Measured.
        plt.figure(figsize= (figure_size_w,figure_size_l))
        ax = plt.subplot()
        ax.scatter(y_test, predicted, edgecolors=(.5,1,0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
        plt.title('{} BU Measured vs Predicted'.format(BU))#,fontsize = 14)
        anchored_text = AnchoredText('R-squared = {:.2f}\nAverage baseline error: {:.2f}\nMean Absolute Error: {:.2f}'.format(r2,np.mean(baseline_errors),np.mean(errors)), loc=2,prop=dict(size=16))
        ax.add_artist(anchored_text)
#        ax.text(1, 13,'R-squared = {:.2f}'.format(r2), fontsize=14)
#        ax.text(1, 12.5,'Average baseline error: {:.2f}'.format(np.mean(baseline_errors)), fontsize=14) 
#        ax.text(1, 12,'Mean Absolute Error: {:.2f}'.format(np.mean(errors)), fontsize=14) 
        ax.set_xlabel('Measured')#, fontsize=14)
        ax.set_ylabel('Predicted')#, fontsize=14)
        plt.tight_layout()
#        plt.show()
        filename = 'Presentation/' + BU + 'image1.png'
        plt.savefig(filename)
        plt.clf()
        #Plot Feature Importances
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title('{} BU Feature Importances'.format(BU))
        plt.barh(range(len(indices)), importances[indices], color='r', align='center')
        plt.rc('ytick')#, labelsize=8)
        plt.yticks(range(len(indices)), Predictors)
        plt.xlabel('Relative Importance')
        plt.tight_layout()
#        plt.show()
        filename = 'Presentation/' + BU + 'image2.png'
        plt.savefig(filename)
        plt.clf()
#        #Scatter plot of actual values with line of prediction from model.
        X = pd.DataFrame(dfBU, columns=Predictors)
        predicted = model.predict(X)
        fig, ax1 = plt.subplots(figsize= (figure_size_w,figure_size_l))
        plt.title('{} BU Summary'.format(BU))
        color = 'red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Incidents', color=color)
        ax1.plot(dfBU['DateKey'], dfBU[To_Predict], 'o', color=color, label='Forward Rolling 14 Day Incident Count')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(np.nan, '-b', label = 'Backwards Rolling 45 Day Oberservation Count')
        ax1.legend(loc='best')
        color = 'darkgreen'
        ax1.plot(dfBU['DateKey'], predicted, color=color, label='Model Prediction')
        ax1.legend(loc='lower right')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'b'
        ax2.set_ylabel('Observations', color=color)  # we already handled the x-label with ax1
        ax2.plot(dfBU['DateKey'], dfBU['event_Observation_Rolling45'], '-', color=color, label='Backwards Rolling 45 Day Oberservation Count')
        ax2.tick_params(axis='y', labelcolor=color)  
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
#        plt.show()
        filename = 'Presentation/' + BU + 'image3.png'
        fig.savefig(filename)
        plt.clf()
#        #Used below code to output resutls for simulation verification.
#        results = pd.DataFrame(dfBU, columns=['BusinessUnit','DateKey','event_Incident_Prediction_Rolling'])
#        predicted = pd.Series(predicted)
#        results['event_Incident_Prediction_Rolling_Prediction'] = predicted.values
#        results.to_csv('results2.csv', mode='a', header=False)



#We have excluded "Material Release" from the dataset. 
#Consider Poisson Regression. #Seems very difficult to implement. Add this as a how to possibly improve note.