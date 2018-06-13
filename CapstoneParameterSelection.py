#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:59:47 2018

@author: evan
"""
import pandas as pd
filename = 'CapstoneParameterTuneResults.csv'
df = pd.read_csv(filename,
        usecols=[i+2 for i in range(13)], 
        index_col=[0,1,3],#['BusinessUnit','ModelType','PredictionBucket'],
        sep=',',
        header=-1,
        names=['BusinessUnit','ModelType','r2','PredictionBucket','Predictor1','Predictor2','Predictor3','Predictor4','Predictor5'
               ,'Predictor6','Predictor7','Predictor8','Predictor9'
#               ,'Predictor10'
#               ,'Predictor11'
               #,'Predictor12','Predictor13'
#               ,'Predictor14','Predictor15'
#               ,'Predictor16','Predictor17','Predictor18','Predictor19'
#               ,'Predictor20'
#               ,'Predictor21'
               ]
        )

df.reset_index()
#useful_columns = ['BusinessUnit','ModelType','r2','PredictionBucket']
#df[useful_columns]
#print(df.info())
#print(df.describe())
#print(df.Date.min)
#print(df.head(20))
#print(df.tail(20))


ndf = df.assign(r2Min = df['r2'],r2Max=df['r2']).groupby(['BusinessUnit','ModelType','PredictionBucket']).agg({'r2Min':'min','r2Max':'max'})
print(ndf)

#print(df.loc[(df['r2'] == .48)])
#print(df.loc[(df['r2'] == .42) & (df['BusinessUnit'] == 'West')])