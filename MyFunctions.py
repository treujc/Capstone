"""
Created on Thu May 24 07:27:27 2018

@author: evan

File that contains useful functions.
"""
import pandas as pd
from datetime import timedelta


def create_dates_dataframe(start_date,end_date):
    """
    Creates a dataframe called dfDates by passing a start_date and end_date
    """
    DateList = [start_date]
    while max(DateList) < end_date:
        DateKey = max(DateList) + timedelta(days=1)
        DateList.append(DateKey)
    DateList.sort() 
    dfDates = pd.DataFrame(pd.to_datetime(DateList), columns = ['DateKey'])
    return dfDates


def dataframe_crossjoin(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res