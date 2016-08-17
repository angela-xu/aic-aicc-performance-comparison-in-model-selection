# AIC vs AICc: Performance Comparison in Model Selection
## Introduction
   In this repository, simulations were used to compare AIC and AICc on their performance in regression model selection.
   * The program is written in Python.
   * Two files are included: 1) aic.py - simulation code; 2) AIC_vs_AICc.png - sample visualization of model selection performance.

## Backgound
* To select an approximating model with least discrepancy toward its true model is one of the primary criteria when researchers do model selection. For years, Akaike information criterion (AIC), which was designed to be an approximately unbiased estimator
of the expected Kullback-Leibler information of a fitted model, is one of the leading model selection methods. And the minimum-AIC
criterion produces a selected model that is close to the best choice. However, this selection criterion tend to overfit severely when the sample size is small. To deal with this problem, Clifford M. Hurvich
and Chih-Ling Tsai derived a bias correction to the Akaike information criterion (AIC) for regression models (Clifford M. Hurvich and 
Chih-Ling Tsai, 1989). The derived corrected method, called AICc, is of particular use when the sample size is small. And it is shown 
to be asymptotically efficient if the true model is infinite dimensional and to provide better model order choices than any other asymptotically
efficient method if the true model is of finite dimension. 
* This project was determined to compare AIC and AICc on their performance in regression model selection with much greater runs of simulations as well as to compare their performance under situations where
model assumptions are not met.

## Visualization & Results
![aic_vs_aicc](https://cloud.githubusercontent.com/assets/19921232/17572898/db8a34c2-5f0b-11e6-9b5b-19789a8b8c6b.png)

* The true model order is 3.
* AIC selection distribution
[   0.    2.    2.  464.  207.  325.    0.    0.    0.    0.]
* AICc selection distribution
[   0.  112.   45.  837.    6.    0.    0.    0.    0.    0.]
* Conclusion: When sample size is small, AICc has significantly better performance than AIC in choosing the correct model order. When sample size is large, AICc and AIC have no difference in model selection. When the normality assumption of model is not met, AICc performs better than AIC. Thus, AICc should be used routinly in place of AIC for regression model selection.  

## Installation and Usages
* Downlaod and run aic.py.
* Parameters could be customized.  
