# WISPER_crosscal_models

Temporary readme file will be updated.

Tune regression models for cross calibration of sensor measurements (isotope analyzers) during the NASA ORACLES project. 

This project included three sensors taking simultaneous measurements (time series) about 90% of the time. Two of the 
instruments do not have calibration information and must be cross-calibrated to the third sensor, which is fully calibrated. 

Using two sensor features (time series) and their cross-term, a polynomial regression model between pairs of intruments were 
tuned, acheiving 70% improvement in RMS error between sensors. 

![Model assessment](https://github.com/DeanHenze/WISPER_crosscal_models/blob/main/figure_model_assessment.png)

$so_{cal} = \Sigma_{i}^{n1}[c_{1i}x_{1}^{i}] + \Sigma_{i}^{n2}[c_{2i}x_{2}^{i}] + \Sigma_{i}^{n12}[c_{12i}x_{12}^{i}] + error$
