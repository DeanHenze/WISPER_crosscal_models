# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:47:32 2021

@author: Dean

Run this script to identify cross-calibration models and tune model parameters 
for Pic2 vs Pic1 water concentration and isotope ratios. 

Separate models are tuned for ORACLES 2017 and 2018. For each year, all dates 
where there is good WISPER data are used; For 2017, Aug. 12th and 13th, 2017, 
Mako's calibration was clearly off and so these flights are omitted.

For water concentration, the model is assumed to be a line 
passing through the origin and therefore only the slope needs to be tuned.

For the isotope ratios, the cross-cal model is assumed to be a polynomial 
function of Pic2-measured water concentration (q), the respective isotope 
ratio delta value (del), and their cross term. The highest power for each of 
these three terms is identified by minimizing the Beyesian Information 
Criterion. 

The water concentration slopes and the linear coefficients for the optimized 
polynomial models are saved as *.csv files in this folder. Publication-ready 
figures are also generated and saved in this folder.

Function list (ordered by relevance)
=============

get_fits: 
    This fxn is run during a call to main, and calls all fxns below either 
    directly or indirectly. Gets parameter fits and figures for both years.

get_fits_singleyear: 
    Called by 'get_fits()'.
    
qxcal_modelfit:
    Tune the slope for a water concentration linear model. 
    
isoxcal_modelfit:
    Tune the coefficients for terms in an isotope ratio polynomial model with 
    predictor variables q, del, and q*del.
    
get_poly_terms:
     Generates a pd.DataFrame of all needed powers of predictor vars. Used by 
     'isoratioxcal_modelfit()'.
     
model_isoxcal:
    Returns predictions for an isotope ratio cross-calibration model. E.g. 
    once a candidate polynomial model has been identified and parameters 
    tuned using 'isoxcal_modelfit()', this function will use the candidate 
    function to generate cross-calibrated pic2 measurements.

draw_fitfig, model_residual_map: 
    The two functions used to make the figures. 'draw_fitfig' is the main 
    function. 'model_residual_map' generates a 2D map of model residuals 
    that can be used as contour plots.

get_wisperdata:
    Returns all WISPER data for a single ORACLES year.
"""


# Built in:
import os
import itertools

# Third party:
import numpy as np # 1.19.2
import matplotlib.pyplot as plt # 3.3.2
import pandas as pd # 1.1.3
import statsmodels.api as sm # 0.12.0

# My modules:
import oversampler
import qxcal_model
import isoxcal_model



# ORACLES flight dates where WISPER took good data:
dates2017_good = ['20170812','20170813','20170815','20170817','20170818',
                  '20170821','20170824','20170826','20170828','20170830',
                  '20170831','20170902']
dates2018_good = ['20180927','20180930','20181003','20181007','20181010',
                  '20181012','20181015','20181017','20181019','20181021',
                  '20181023']


def get_wisperdata(year):
    """
    Load all WISPER (q,dD,d18O) data for either 2017 or 2018 with good data 
    and average the data into 8 second blocks before returning. 
    Return as a pandas df.
    
    year: str, '2017' or '2018'.
    """
    # Get paths to all data files for the input year:
    if year=='2017':
        dates_good = dates2017_good
    elif year=='2018':
        dates_good = dates2018_good
 
    path_data_dir = r"./sensor_data/"   
    fnames = ['WISPER_pic1cal_%s.ict' % d for d in dates_good]
    paths_data = [path_data_dir+f for f in fnames]
    
    
    # Loop through each date and append data to a single pandas df:
    columns = ['h2o_tot1','h2o_tot2','dD_tot1',
               'dD_tot2','d18O_tot1','d18O_tot2'] # Keep these data columns.
    wisper = pd.DataFrame({}, columns=columns) # Append all data here.
    for p in paths_data:
        data_temp = pd.read_csv(p) # Load.
        data_temp.replace(-9999, np.nan, inplace=True)
        # Average data into 8 s blocks before appending:
        data_blocked = data_temp.groupby(lambda x: np.round(x/8)).mean()
        wisper = wisper.append(data_blocked[columns], ignore_index=True)
    
    return wisper.dropna(how='any') # Drop missing values



def get_fits_singleyear(year, wisperdata):
    """
    Get cross-cal models and fit parameters for water concentration and each 
    isotope ratio for a single ORACLES year. Return as a dict of pandas.Series 
    objects.
    
    Polynomial form of model for isotope ratios is identified by minimizing 
    the Beyesian Information Criterion.
    
    year: str, '2017' or '2018'.
    
    wisperdata: pandas.DataFrame. Contains all WISPER data for the year.
    """            
    print("****************************************************\n"
      "Cross-calibration fit parameters for ORACLES "+year+"\n"
      "****************************************************")
    

    ## Fitting humidity is straightforward:
    ##-----------------
    model_q = qxcal_model.fit(wisperdata, 'h2o_tot1', 'h2o_tot2')
    #print(model_q.summary())
    print('q\n===')
    print('R2 = %f' % model_q.rsquared)


    ## Fitting the iso ratios requires polynomial model selection:
    ##-----------------
    def polyord_minBIC(wisperdata, iso):
        """
        Using min Bayesian info criterion (BIC) to determine highest power 
        (up to 5) of each predictor var and crossterm, for the chosen 
        isotopologue. Returns a 3-tuple of ints, where each is the highest 
        power to raise the predictor vars: logq, iso, and logq*iso. iso is 
        either 'dD' or 'd18O':
        """
        # Cartesian product of all poly orders up to 5:
        nord_list = list(itertools.product(range(1,6), range(1,6), range(1,6)))
        bic_list = []
        for nord in nord_list:
            model = isoxcal_model.fit(wisperdata, iso, nord) # Statsmodels results.
            bic_list.append(model.bic) # Append this run's BIC.
        # Combo of poly orders with the minimum BIC:
        return nord_list[np.argmin(bic_list)]

    # Find optimal polynomial orders for each iso ratio. Then re-run fit with 
    # those poly orders:
    nord_dD = polyord_minBIC(wisperdata, 'dD')
    nord_d18O = polyord_minBIC(wisperdata, 'd18O')
    model_dD = isoxcal_model.fit(wisperdata, 'dD', nord_dD)
    model_d18O = isoxcal_model.fit(wisperdata, 'd18O', nord_d18O)
    
    #print(model_dD.summary())
    #print(model_d18O.summary())
    print('\ndD\n===')
    print('nord = % s' % str(nord_dD))
    print('R2 = %f' % model_dD.rsquared)    
    print('\nd18O\n====')
    print('nord = % s' % str(nord_d18O))
    print('R2 = %f' % model_d18O.rsquared)


    ## Return parameter fits:
    return {'q':model_q.params, 'dD':model_dD.params, 'd18O':model_d18O.params}



def get_fits():
    """
    Get cross-calibration formula fit parameters for water concentration and 
    both isotopologues for both the 2017 and 2018 ORACLES years.
    """  
    """
    ## Check that all WISPER files with calibrated Pic1 data are in the 
    ## necessary directory, otherwise run calibration script to get them:
    ##-----------------        
        # 'paths_data' should be the paths of all the files if they exist:
    datesall_good = (pic1cal.dates2017_good + 
                     pic1cal.dates2018_good) # All relevant P3 flight dates.        
    path_data_dir = pic1cal.path_pic1caldir # directory with data files.
    fnames = ['WISPER_pic1cal_%s.ict' % d for d in datesall_good]
    paths_data = [path_data_dir+f for f in fnames]

    print("Checking that all WISPER 2017 and 2018 pic1-calibrated files "
          "exist. For any that don't, running code to get calibrated files.")        
    for i in range(len(datesall_good)):
        if os.path.isfile(paths_data[i]):
            continue
        else:
            pic1cal.calibrate_20172018_file(datesall_good[i])
    print("All files now exist, good to start cross-calibration fits.")
    """    
    
    ## Fit parameters for each year:
    ##-----------------
    fitparams_2017 = get_fits_singleyear('2017', get_wisperdata('2017'))
    fitparams_2018 = get_fits_singleyear('2018', get_wisperdata('2018'))
    
    
    ## Save H2O xcal fit results to this folder:
    ##-----------------
    slope2017 = fitparams_2017['q']['h2o_tot2']
    slope2018 = fitparams_2018['q']['h2o_tot2']
    h2o_xcal_df = pd.DataFrame({'year':['2017','2018'], 
                                'slope':[slope2017,slope2018]}, 
                               columns=['year','slope'])
    h2o_xcal_df.to_csv("h2o_xcal_results.csv", index=False)
    

    ## Save H2O xcal fit results to this folder:
    ##-----------------
    def isoratio_xcal_to_csv(fitparams_s, fname):
        # fitparams_s: fit results as pd.Series.
        fitparams_df = pd.DataFrame({'predictor_var':fitparams_s.index,
                                     'coeff':fitparams_s.values},
                                    columns=['predictor_var','coeff'])
        fitparams_df.to_csv(fname, index=False)
    
    isoratio_xcal_to_csv(fitparams_2017['dD'], "dD_xcal_results_2017.csv")
    isoratio_xcal_to_csv(fitparams_2017['d18O'], "d18O_xcal_results_2017.csv")
    isoratio_xcal_to_csv(fitparams_2018['dD'], "dD_xcal_results_2018.csv")
    isoratio_xcal_to_csv(fitparams_2018['d18O'], "d18O_xcal_results_2018.csv")
    

if __name__ == '__main__':
    get_fits()