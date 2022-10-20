# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:45:12 2022

@author: Dean
"""


import statsmodels.api as sm # 0.12.0


def modelfit(df, k_s1, k_s2):
    """
    Model fit parameters for vapor concentration of sensor 2 vs sensor 1. 
    
    Model is a simple linear relationship of sensor 1 measurements as a 
    function of sensor 2. Line is constrained to pass through the origin. 
    
    Inputs
    ------
    data: pandas.Dataframe 
        Contains sensor measurements.
    k_s1, k_s2: str's
        Dataframe keys for sensor 1 and 2, respectively.
    
    """
    return sm.OLS(df[k_s1], df[k_s2], missing='drop').fit()