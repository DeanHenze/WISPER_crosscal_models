# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:20:42 2022

@author: Dean
"""



# Third party:
import numpy as np # 1.19.2
import pandas as pd # 1.1.3
import statsmodels.api as sm # 0.12.0

    

def fit(data, iso, nord):
    """
    Determine polynomial fit for isotope ratio cross-calibration data.
    
    Fit Pic1 isotope measurements to a polynomial of Pic2 log(q) and isotope 
    measurements (either dD or d18O), including an interaction term. Use a 
    weighted linear regression from the statsmodels package (weights are the 
    water concentrations). 
    
    Inputs
    ------
    data: pandas.DataFrame. 
        Sensor measurements. 
    
    iso: str.
        The isotope to get cross-cal formula for (either 'dD' or 'd18O'). 
        
    nord: is 3-tuple of ints. 
        The highest power to include for each of the predictor variables logq, 
        isotope-ratio, and their crossterm (in that order).
        
    Returns
    -------
    A statsmodels RegressionResults object (model parameters and fit metrics).
    """
    # Pandas df of predictor vars:
    interterm = np.log(data['h2o_tot2'])*data[iso+'_tot2'] # Interaction term.
    predictvars = pd.DataFrame({'logq':np.log(data['h2o_tot2']),
                                iso:data[iso+'_tot2'],
                                'logq*'+iso:interterm})
    
    # Get all desired powers of each predictor var; add as new df columns:
    pwr_range = {'logq':(0,nord[0]), iso:(0,nord[1]), 
                 'logq*'+iso:(0,nord[2])}
    predictvars_poly = get_poly_terms(predictvars, pwr_range)
    
    # Return model fit:
    model = sm.WLS(
        data[iso+'_tot1'], predictvars_poly, 
        missing='drop',# weights=data['h2o_tot2']
        )
    return model.fit()

    

def predict(predvars, pars):
    """
    The isotope cross-calibration model (results of the polynomial fit). Takes 
    in model parameters and data for the predictor variables (logq, iso, 
    logq*iso), and returns calibrated isotope ratios. iso is either 'dD' or 
    'd18O'.
    
    predvars: dict-like.
        Contains values of the predictor vars (arrays of same length).
        
    pars: (dict-like). 
        Fit parameters. The dict keys need to be of the form 'var^n' where n 
        is the power of the term in the model and 'var' is the key for the 
        appropriate variable in 'data'.
    """
    terms = []
    
    for k in pars.keys():
        if k=='const':
            terms.append(pars[k]*np.ones(np.shape(predvars[list(predvars.keys())[0]])))
        else:
            pvar, power = k.split('^') # Predictor var name and power it's raised to.
            terms.append(pars[k]*predvars[pvar]**int(power))
    
    return np.sum(terms, axis=0)



def get_poly_terms(predictvars, pwr_range):
    """
    Produces powers of predictor vars for a candidate model to use with 
    statsmodels. 
    
    predictvars: pandas.DataFrame.
        Contains predictor vars for model.
    pwr_range: dict.
        Keys are a subset of the columns in predictvars. Elements are 
        2-tuples for min, max powers for that key. Powers can be negative.
    """
    modelvars = pd.DataFrame({}) # Will hold all vars to use with model.
    
    for key in pwr_range.keys():
        # Powers (excluding 0) of predictor var to try:
        powers = list(range(pwr_range[key][0],pwr_range[key][1]+1))
        if 0 in powers: del powers[powers.index(0)]
        
        for i in powers: # Compute and append powers:
            modelvars[key+'^%i' % i] = predictvars[key]**i
            
    # Add constant offset var:
    modelvars['const'] = np.ones(len(predictvars))
            
    return modelvars