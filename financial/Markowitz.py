# -*- coding: utf-8 -*-
"""
Script to show the Markowitz bullet working principle with two assets
and different correlation coefficients

Created on Wed Oct 14 13:03:05 2015

@author: Andreu Mora
"""

import numpy as np
import matplotlib.pyplot as plt

x = [0.5, 0.5]

mu = np.array([3./100, 1./100])
sigma = np.array([15./100, 5./100])
label = ['Asset A', 'Asset B']
rf = 0.5/100

f = plt.figure()

w = np.zeros((100,2))
w[:,0] = np.linspace(-0.5,1.5,w.shape[0])
w[:,1] = 1 - w[:,0]

corr_vec =  [-1, -0.5, 0, 0.5, 1]
for corr in corr_vec:
    
    # Processing
    cov = [[sigma[0]**2, corr*sigma[0]*sigma[1]],[corr*sigma[0]*sigma[1], sigma[1]**2]]
    
    ret_out = np.dot(w, mu)
    var = ret_out*0
    for i in range(0,w.shape[0]):
        var[i] = np.dot(np.dot(w[i], cov), w[i])
    std_out = np.sqrt(var)
    plt.plot(std_out, ret_out)
    
    # Compute the minimum variance portfolio
    inv_cov = np.linalg.inv(cov)
    Sigma_1 = np.dot(inv_cov, [1,1])
    m = Sigma_1 / np.dot([1,1], Sigma_1)
    ret_m = np.dot(m, mu)
    var_m = np.dot(np.dot(m,cov),m)
    std_m = np.sqrt(var_m)
    plt.plot(std_m, ret_m, '*')
    
    # Compute the tangency portfolio
    Sigma_rf = np.dot(inv_cov, mu-rf)
    tan = Sigma_rf / np.dot([1,1], Sigma_rf)
    ret_tan = np.dot(tan, mu)
    var_tan = np.dot(np.dot(tan,cov),tan)
    std_tan = np.sqrt(var_tan)
    plt.plot([0, std_tan], [rf, ret_tan], '--k')

plt.grid()
for i in [0,1]:
    #plt.plot(sigma[i], mu[i],'b*')
    plt.text(sigma[i], mu[i], label[i])
    plt.hold()
plt.show()
#plt.legend(corr_vec, loc=2)
plt.title('Markowitz bullet for two assets with tangency and minimum variance')