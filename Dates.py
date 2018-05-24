#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 07:27:27 2018

@author: evan
"""
import pandas as pd
from datetime import timedelta


def create_dates_dataframe(start_date,end_date):
    DateList = [start_date]
    while max(DateList) < end_date:
        DateKey = max(DateList) + timedelta(days=1)
        DateList.append(DateKey)
    DateList.sort() 
    dfDates = pd.DataFrame(pd.to_datetime(DateList), columns = ['DateKey'])
    return dfDates