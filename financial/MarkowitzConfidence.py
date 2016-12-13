# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:03:05 2015

Example script to illustrate 

@author: Andreu Mora
"""

import numpy as np
import matplotlib.pyplot as plt

def buildCovMat (sigma, corr):
    return [[sigma[0]**2, corr*sigma[0]*sigma[1]],[corr*sigma[0]*sigma[1], sigma[1]**2]]
    
def evaluatePortfolio (w, mu, cov):
    ret_out = np.dot(w, mu)
    var = ret_out*0
    if type(var)==np.ndarray:
        for i in range(0,w.shape[0]):
            var[i] = np.dot(np.dot(w[i], cov), w[i])
    else:
        var = np.dot(np.dot(w, cov), w)
    std_out = np.sqrt(var) 
    return ret_out, std_out
    
def computeMinVarPortfolio (cov):
    inv_cov = np.linalg.inv(cov)
    Sigma_1 = np.dot(inv_cov, [1,1])
    return Sigma_1 / np.dot([1,1], Sigma_1)
    
def computeTangencyPortfolio (cov, mu, rf):
    inv_cov = np.linalg.inv(cov)
    Sigma_rf = np.dot(inv_cov, mu-rf)
    return Sigma_rf / np.dot([1,1], Sigma_rf)
    
mu = np.array([3./100, 1./100])
sigma = np.array([15./100, 5./100])
label = ['Asset A', 'Asset B']
rf = 0.5/100
corr = 0.1

conf_mu = np.array([[2.5/100, 3.25/100], [0.85/100, 1.1/100]])
conf_sigma = np.array([[13.5/100, 16.5/100],[4.4/100, 5.7/100]])
conf_corr = [-0.13, 0.25]


f = plt.figure()

w = np.zeros((100,2))
w[:,0] = np.linspace(-0.5,1.5,w.shape[0])
w[:,1] = 1 - w[:,0]

# Processing
cov = buildCovMat(sigma, corr)
ret, std = evaluatePortfolio(w, mu, cov)
plt.plot(std, ret, '-k')
    
# Compute the minimum variance portfolio
m = computeMinVarPortfolio(cov)
ret_m, std_m = evaluatePortfolio(m, mu, cov)
plt.plot(std_m, ret_m, '*')
    
# Compute the tangency portfolio
tan = computeTangencyPortfolio(cov, mu, rf)
ret_tan, std_tan = evaluatePortfolio (tan, mu, cov)   
plt.plot([0, std_tan], [rf, ret_tan], '--k')

plt.grid()
for i in [0,1]:
    #plt.plot(sigma[i], mu[i],'b*')
    plt.text(sigma[i], mu[i], label[i])
    plt.hold()
plt.show()
#plt.legend(corr_vec, loc=2)
plt.title('Markowitz bullet for two assets with tangency and minimum variance')

# Plot the confidence intervals
for i in range(0,conf_mu.shape[0]):
    # Plot for the different correlation coefficients
    for j in range(0, len(conf_corr)):
        # Compute the portfolio for this combination
        cov = buildCovMat(conf_sigma[:,i], conf_corr[j])
        ret_out, std_out = evaluatePortfolio(w, conf_mu[:,i], cov)
        plt.plot(std_out, ret_out, '-')
        tan = computeTangencyPortfolio(cov, conf_mu[:,i], rf)
        ret_tan, std_tan = evaluatePortfolio(tan, conf_mu[:,i], cov)
        plt.plot([0, std_tan], [rf, ret_tan], '--')