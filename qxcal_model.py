# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:45:12 2022

@author: Dean

Model fit parameters for vapor concentration of sensor 2 vs sensor 1. 
"""


import statsmodels.api as sm # 0.12.0


def fit(data, k_s1, k_s2):
    """    
    Model is a simple linear relationship of sensor 1 measurements as a 
    function of sensor 2. Line is constrained to pass through the origin. 
    
    Inputs
    ------
    data: pandas.Dataframe 
        Sensor measurements.
    k_s1, k_s2: str's
        Dataframe keys for sensor 1 and 2, respectively.
        
    Returns
    -------
    A statsmodels RegressionResults object (model parameters and fit metrics).
    """
    return sm.OLS(data[k_s1], data[k_s2], missing='drop').fit()