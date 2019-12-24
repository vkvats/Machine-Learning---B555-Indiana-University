#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# In[6]:


def MSE_test_set(design_matrix_train, response_variable_train, design_matrix_test, response_variable_test,value):
    
# Model parameter
        
    phi_transpose_t = design_matrix_train.transpose().dot(response_variable_train)
    phi_transpose_phi = design_matrix_train.transpose().dot(design_matrix_train)
    indentity = np.identity(len(phi_transpose_phi))  
    model_parameter =np.linalg.inv(np.add(indentity.dot(value),phi_transpose_phi)).dot(phi_transpose_t)
#     print(model_parameter)
    
    
    phi_w_t_test = np.subtract(design_matrix_test.dot(model_parameter),response_variable_test)
    phi_w_t_sq_test = 0
    MSE_test0 = list()
    for x in np.nditer(phi_w_t_test):
        phi_w_t_sq_test += x**2
    MSE_test0.append(phi_w_t_sq_test/len(design_matrix_test))
    return MSE_test0


def alpha_beta_calculation(design_matrix_train, response_variable_train, alpha_init, beta_init): 
    

    # SN_inverse= alpha*I + beta*phi_transpose*phi
    
    phi_transpose_phi = (design_matrix_train.transpose().dot(design_matrix_train))
       
    eigenvalue_init = np.linalg.eigvalsh(phi_transpose_phi)
    
    beta = beta_init
    alpha = alpha_init
    
    n_iteration= 0
    
    #start of loop
    for itr in range(100):
#         n_iteration += 1
        alpha_old = alpha
        beta_old = beta

        # find MN and SN using alpha and beta

        alpha_I = (np.identity(len(phi_transpose_phi)))*alpha
        beta_phi_transpose_phi = beta*(phi_transpose_phi)
        SN_inverse = np.add(alpha_I, beta_phi_transpose_phi)
        SN = np.linalg.inv(SN_inverse)

        phi_transpose_t = design_matrix_train.transpose().dot(response_variable_train)
        MN = (SN.dot(phi_transpose_t))*beta
    #     print(MN)
        MN_transpose_MN = np.sum( MN**2) # MN.transpose().dot(MN)


        # Find new alpha and beta using MN and SN
        
        eigenvalues = eigenvalue_init * beta
        gamma = np.sum(eigenvalues / (eigenvalues + alpha)) # change in eigenvalues is not accounted for yet
        alpha = gamma/MN_transpose_MN # new alpha
        error_term = np.subtract(response_variable_train,design_matrix_train.dot(MN))
        error_term_squared = np.sum(error_term**2)[0]
        beta = (len(response_variable_train) - gamma)/error_term_squared # new beta
#         print("iteration", itr)
#         print("gamma", gamma)
#         print("alpha", alpha)
#         print("beta", beta)
        if np.isclose(alpha_old, alpha, atol=1e-6) and np.isclose(beta_old, beta, atol= 1e-6):
            return alpha, beta, MN, SN, itr
    
    # loop back to find New MN and SN
 


# In[9]:


data_files = { "data 100_10": ("train-100-10.csv", "trainR-100-10.csv", "test-100-10.csv", "testR-100-10.csv"),
    "data 100_100": ("train-100-100.csv", "trainR-100-100.csv", "test-100-100.csv", "testR-100-100.csv"),
    "data 1000_100": ("train-1000-100.csv", "trainR-1000-100.csv", "test-1000-100.csv", "testR-1000-100.csv"),
    "data crime": ("train-crime.csv", "trainR-crime.csv", "test-crime.csv", "testR-crime.csv"), 
    "data wine": ("train-wine.csv", "trainR-wine.csv", "test-wine.csv", "testR-wine.csv") }

output = dict()

for key,data in data_files.items():
    design_matrix_train = pd.read_csv(data[0], header = None)
    response_variable_train = pd.read_csv(data[1], header = None)
    design_matrix_test = pd.read_csv(data[2], header = None)
    response_variable_test = pd.read_csv(data[3], header = None)
    start_time = time.time()

    
    alpha, beta, MN, SN, iteration = alpha_beta_calculation(design_matrix_train, response_variable_train, 1, 5)

    lambda_value = alpha/beta

    MSE_test = MSE_test_set(design_matrix_train, response_variable_train, design_matrix_test, response_variable_test,lambda_value)
    
    output[key]= {"Test MSE of {}".format(key):round(MSE_test[0],4), 
                  "alpha {}". format(key): round(alpha,4),
                  "beta {}". format(key): round(beta,4),
                  "Lamda {}".format(key): round(lambda_value,4),
                  "No of iteration {}".format(key): iteration,
                 "Run time of {}".format(key): round(time.time()- start_time,3)}


# In[10]:


print(output)


# In[ ]:





# In[ ]:




