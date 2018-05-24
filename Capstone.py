import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, Binarizer
import MyFunctions

#file locations
current_dir = os.path.realpath(".")
data_folder = os.path.join(current_dir,'data')
datafile = os.path.join(data_folder,'CapstoneData.csv')
dataheaderfile = os.path.join(data_folder,'CapstoneDataHeaders.csv')

#Read in data files.
df_headers = list(pd.read_csv(dataheaderfile))
df = pd.read_csv(datafile,names = df_headers, low_memory = False)

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



#df_trimmed.info()
#df_trimmed.head()
df_trimmed.event.unique()
# df_trimmed.eventClassification.unique()


##categories can be considered counts over time. Which could then become ratios.
categories = ['event','eventClassification']#,'companyInvolved','operationOrDevelopment','jobTypeObserved','stopJob','immediateActionsTaken','rigInvolved']
BUMatrix = pd.get_dummies(df_trimmed,columns = categories)


#BUMatrix.head()
useful_columns = ['BusinessUnit','eventOccurredDate','event_Observation','event_Incident']
BUMatrix = BUMatrix[useful_columns]


BUMatrix['eventOccurredDate'] = pd.to_datetime(df['eventOccurredDate']).dt.date
#BUMatrix = BUMatrix.groupby(BUMatrix.eventOccurredDate).sum()
BUMatrix = BUMatrix.groupby([BUMatrix.eventOccurredDate,BUMatrix.BusinessUnit]).sum()
BUMatrix = BUMatrix.reset_index()
BUMatrix.sort_values(by = ['eventOccurredDate'])

BUList = BUMatrix.BusinessUnit.unique()
#Remove Central Support due to
BUList = BUList[BUList != 'Central Support'] 

#Create dfDates dataframe based on max and min values in main dataframe.
end_date = BUMatrix['eventOccurredDate'].max()
start_date = BUMatrix['eventOccurredDate'].min()
dfDates = MyFunctions.create_dates_dataframe(start_date,end_date)


def BUPlot(BU,i):
    # plt.subplot(3,2,i+1)
    BUMatrix[BUMatrix['BusinessUnit'] == BU].plot(x='eventOccurredDate', y=['event_Observation','event_Incident'],style=".",figsize=(15,15))

# for BU in BUList:
#     BUPlot(BU)

# =============================================================================
# for i, BU in enumerate(BUList):
#     BUPlot(BU,i)
# plt.show()
# =============================================================================
#BUMatrix.plot(kind='scatter',x='', y='event_Observation')
# BUMatrix.plot(x='eventOccurredDate', y=['event_Observation','event_Incident'],style=".",figsize=(15,15))
# plt.show()
# from pandas.plotting import scatter_matrix
# scatter_matrix(BUMatrix)

# from pandas.plotting import scatter_matrix
# print(scatter_matrix(BUMatrix))
######Fill in the Date values so you have 1 date for every day. Do a count of each incident type over a set window.

dfBUList = pd.DataFrame(BUList,columns = ['BusinessUnit'])
dfCounts = MyFunctions.dataframe_crossjoin(dfDates, dfBUList)

print(dfCounts)
