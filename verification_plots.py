# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:47:54 2022

@author: Dean
"""


# Third party:
import numpy as np # 1.19.2
import matplotlib.pyplot as plt # 3.3.2
import pandas as pd # 1.1.3
import statsmodels.api as sm # 0.12.0

# Local code:
import oversampler
import qxcal_model
import isoxcal_model
import pic2_xcal_fits



def scatter_with_modelfit(year, wisperdata, pars_dD, pars_d18O):
    """
    Publication read figure. Colored scatter plot of cross-calibration data 
    and 2D colored-contour maps of the polynomial fit, for both dD and d18O. 
    Figures are saved in this folder.
    """
    fig = plt.figure(figsize=(6.5,2.5))
    ax_D = plt.axes([0.125,0.2,0.29,0.75])
    cbax_D = plt.axes([0.435,0.2,0.02,0.625])
    ax_18O = plt.axes([0.62,0.2,0.29,0.75])
    cbax_18O = plt.axes([0.93,0.2,0.02,0.625])
    
    ## Some year-dependent contour/colormapping values:
    if year=='2017':
        # Min/max values for colormap renormalization:
        vmin_D = -650; vmax_D = 0; vmin_18O = -80; vmax_18O = 0
        # Contour levels for plotting model output:
        clevs_D = np.arange(-600,0,50); clevs_18O = np.arange(-80,0,5)
    if year=='2018':
        vmin_D = -200; vmax_D = -40; vmin_18O = -30; vmax_18O = -8
        clevs_D = np.arange(-200,0,15); clevs_18O = np.arange(-30,0,2)

    ## WISPER data scatter plots:
    ##-------------------------------------------------------------------------
        # Thin out the wisper data for better visuals:
    wisperthin = wisperdata.iloc[np.arange(0,len(wisperdata),10)]
        
    s_D = ax_D.scatter(np.log(wisperthin['h2o_tot2']), wisperthin['dD_tot2'], 
                       c=wisperthin['dD_tot1'], vmin=vmin_D, vmax=vmax_D, 
                       s=5)
    s_18O = ax_18O.scatter(np.log(wisperthin['h2o_tot2']), wisperthin['d18O_tot2'], 
                           c=wisperthin['d18O_tot1'], 
                           vmin=vmin_18O, vmax=vmax_18O, s=5)
    ##-------------------------------------------------------------------------

    ## Compute model-fit values and plot as contours:
    ##-------------------------------------------------------------------------
        # Predictor variable values to pass into model:
    if year=='2017':
        logq = np.linspace(np.log(100), np.log(30000), 200)
        logq_grid, dD_grid = np.meshgrid(logq, np.linspace(-450, -30, 100))
        logq_grid, d18O_grid = np.meshgrid(logq, np.linspace(-55, 0, 100))
    if year=='2018':
        logq = np.linspace(np.log(2000), np.log(30000), 200)
        logq_grid, dD_grid = np.meshgrid(logq, np.linspace(-200, -30, 100))
        logq_grid, d18O_grid = np.meshgrid(logq, np.linspace(-55, -30, 100))
    predictorvars = {'logq':logq_grid, 
                     'dD':dD_grid, 
                     'd18O':d18O_grid, 
                     'logq*dD':logq_grid*dD_grid, 
                     'logq*d18O':logq_grid*d18O_grid
                     }

        # Run model:
    modeldata_dD = isoxcal_model.predict(predictorvars, pars_dD)
    modeldata_d18O = isoxcal_model.predict(predictorvars, pars_d18O)
    
        # Contour plots of model output:
    ax_D.contour(logq_grid, dD_grid, modeldata_dD, 
                 levels=clevs_D, vmin=vmin_D, vmax=vmax_D, linewidths=2.5)
    ax_18O.contour(logq_grid, d18O_grid, modeldata_d18O, 
                   levels=clevs_18O, vmin=vmin_18O, vmax=vmax_18O, linewidths=2.5)
    ##-------------------------------------------------------------------------
    
    ## Include contours of model residuals
    ##-------------------------------------------------------------------------
        # Compute root-mean-square deviations:
    if year=='2017':
        reslevs_D = [2,5,15,30]; reslevs_18O = [0.2,0.5,1,2,4]; ffact=1.5
    if year=='2018': 
        reslevs_D = [1,2,5,10,20]; reslevs_18O = [0.2,0.5,1,2]; ffact=4
    
    res_dD = model_residual_map('dD', wisperdata, pars_dD, logq, dD_grid[:,0], ffact=ffact)  
    res_d18O = model_residual_map('d18O', wisperdata, pars_d18O, logq, d18O_grid[:,0], ffact=ffact)  
        
        # Contours:
    rescont_D = ax_D.contour(logq_grid, dD_grid, res_dD, 
                             levels=reslevs_D, colors='black', linewidths=1)
    rescont_18O = ax_18O.contour(logq_grid, d18O_grid, res_d18O, 
                                 levels=reslevs_18O, colors='black', linewidths=1)
    plt.clabel(rescont_D, inline=True, fmt='%i')
    plt.clabel(rescont_18O, inline=True, fmt='%0.1f')
    ##-------------------------------------------------------------------------

    ## Figure axes labels, limits, colobars, ...
    ##-------------------------------------------------------------------------
        # Results axes mods:
    if year=='2017':
        ax_D.set_xlim(4.5, 10.5); ax_D.set_ylim(-400, -20)
        ax_18O.set_xlim(4.5, 10.5); ax_18O.set_ylim(-50, 0)
    if year=='2018':
        ax_D.set_xlim(7.5, 10.5); ax_D.set_ylim(-180, -40)
        ax_18O.set_xlim(7.5, 10.5); ax_18O.set_ylim(-50, -32)

    ax_D.set_xlabel(r'log($q_2$[ppmv])', fontsize=12)
    ax_D.set_ylabel(r'$\delta D_2$'+u'(\u2030)', fontsize=12, labelpad=0)
    ax_18O.set_xlabel(r'log($q_2$[ppmv])', fontsize=12)
    ax_18O.set_ylabel(r'$\delta^{18} O_2$'+u'(\u2030)', fontsize=12, labelpad=0)   
    
        # Colorbar axes mods:
    fig.colorbar(s_D, cax=cbax_D)
    fig.colorbar(s_18O, cax=cbax_18O)
    
    cbax_D.set_title(r'$\delta D_1$'+'\n'+u'(\u2030)', fontsize=10)
    plt.setp(cbax_D.yaxis.get_majorticklabels(), 
             ha="center", va="center", rotation=-90, rotation_mode="anchor")
    
    cbax_18O.set_title(r'$\delta^{18} O_1$'+'\n'+u'(\u2030)', fontsize=10)
    plt.setp(cbax_18O.yaxis.get_majorticklabels(), 
         ha="center", va="center", rotation=-90, rotation_mode="anchor")
    ##-------------------------------------------------------------------------
    
    fig.savefig("pic2_isoratio_xcal_fitresults_%s.png" % year)
    
    
   
def model_residual_map(iso, wisperdata, pars, logq_grid, iso_grid, ffact=1):
    """
    Returns a 2D, q-dD map of residuals for an isotope cross calibration. 
    """
    
    # Get model predictions:
    logq = np.log(wisperdata['h2o_tot2'].values)
    predictorvars = {'logq':logq, 
                     iso:wisperdata[iso+'_tot2'].values, 
                     'logq*'+iso:logq*wisperdata[iso+'_tot2'].values, 
                     }
    modelresults = isoxcal_model.predict(predictorvars, pars)
    # Model residuals:
    res = abs(modelresults - wisperdata[iso+'_tot1'])
    # Get RMSE 2d map using oversampling:
    return oversampler.oversampler(res, logq, wisperdata[iso+'_tot2'], 
                                   logq_grid, iso_grid, ffact=ffact)

    

def rmse_map(res, logq, deliso, logq_grid, deliso_grid, ffact=1):
    """
    Returns a 2D map of RMSE for an isotope cross calibration. 
    """
    
    # Get RMSE 2d map using oversampling:
    mse = oversampler.oversampler(res**2, logq, deliso, 
                                  logq_grid, deliso_grid, ffact=ffact) 
    rmse = mse**0.5
    return rmse


def model_residuals(iso, wisperdata, pars):
    """
    Model residuals from the isotope cross-calibration model predictions. 
    """
    
    # Get model predictions:
    logq = np.log(wisperdata['h2o_tot2'].values)
    predictorvars = {'logq':logq, 
                     iso:wisperdata[iso+'_tot2'].values, 
                     'logq*'+iso:logq*wisperdata[iso+'_tot2'].values, 
                     }
    modelresults = isoxcal_model.predict(predictorvars, pars)
    # Model residuals:
    res = abs(modelresults - wisperdata[iso+'_tot1'])
    return res
    


year = '2018'

wisperdata = pic2_xcal_fits.get_wisperdata(year)

pars_dD = pd.read_csv("dD_xcal_results_%s.csv" % year, index_col='predictor_var')
pars_dD = pars_dD.squeeze()
pars_d18O = pd.read_csv("d18O_xcal_results_%s.csv" % year, index_col='predictor_var')
pars_d18O = pars_d18O.squeeze()

scatter_with_modelfit(year, wisperdata, pars_dD, pars_d18O)