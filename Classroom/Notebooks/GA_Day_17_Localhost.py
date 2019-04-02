# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np


df = pd.read_csv('H-1B_Disclosure_Data_FY17.csv')


drop = df.columns[(df.isna().sum()/df.shape[0])*100 > 30].tolist()

df = df.drop(drop,1)

df = df.drop(['Unnamed: 0','CASE_NUMBER'],1)

cols = list(df.columns)

df[cols] = df[cols].fillna(df.mode().iloc[0])


df = df.drop(['WORKSITE_CITY','WORKSITE_COUNTY','WORKSITE_POSTAL_CODE'],1)

df = df.drop(['EMPLOYER_NAME','EMPLOYER_ADDRESS','EMPLOYER_CITY','EMPLOYER_POSTAL_CODE','EMPLOYER_COUNTRY','EMPLOYER_PHONE','AGENT_ATTORNEY_NAME','SOC_NAME','PW_SOURCE_OTHER'],1)
