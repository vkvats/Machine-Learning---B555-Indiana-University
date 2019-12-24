#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (20,10)
# get_ipython().run_line_magic('matplotlib', 'inline')


# # MSE calculation

# In[3]:


def MSE_train_test(design_matrix_train, response_variable_train, design_matrix_test, response_variable_test):
    
     # Model parameter
        
    phi_transpose_t = design_matrix_train.transpose().dot(response_variable_train)
    phi_transpose_phi = design_matrix_train.transpose().dot(design_matrix_train)
    indentity = np.identity(len(phi_transpose_phi))  
    model_parameter = list()
    for lambda_ in range(150):
        model_parameter.append(np.linalg.inv(np.add(indentity.dot(lambda_),phi_transpose_phi)).dot(phi_transpose_t))
        
    # MSE train

    MSE_train = list()
    for i in range(150):
        phi_w_t_train = np.subtract(design_matrix_train.dot(model_parameter[i]),response_variable_train)
        phi_w_t_sq_train = 0
        for x in np.nditer(phi_w_t_train):
            phi_w_t_sq_train += x**2
        MSE_train.append(phi_w_t_sq_train/len(design_matrix_train))
        
    # MSE Test
    MSE_test = list()
    for i in range(150):
        phi_w_t_test = np.subtract(design_matrix_test.dot(model_parameter[i]),response_variable_test)
        phi_w_t_sq_test = 0
        for x in np.nditer(phi_w_t_test):
            phi_w_t_sq_test += x**2
#             print(phi_w_t_sq_test)
        MSE_test.append(phi_w_t_sq_test/len(design_matrix_test))    
    
    return MSE_train, MSE_test


# In[ ]:





# In[4]:


# plotting the data
def MSE_vs_lambda_plot(MSE_train, MSE_test, key):
    x1 = range(150)
    y1 = MSE_train
    y2 = MSE_test
    a = min(y1)
    b = max(y2)
    d = (max(y2) - min(y2))/6
    plt.plot(x1, y1, label = "MSE Train")
    plt.plot(x1, y2, label = "MSE Test")
    plt.xlabel('Value of lambda')
    plt.ylabel("Mean Squared Error")
    plt.legend(loc=0)
    plt.title("MSE vs lambda for {} data".format(key))
    plt.yticks(np.arange(a, b, d))      
    plt.xticks(range(0,151,10))
    plt.rcParams["figure.figsize"]=(15,7)
    return plt.show()
        


# In[5]:


data_files = { "100_10": ("train-100-10.csv", "trainR-100-10.csv", "test-100-10.csv", "testR-100-10.csv"),
    "100_100": ("train-100-100.csv", "trainR-100-100.csv", "test-100-100.csv", "testR-100-100.csv"),
    "1000_100": ("train-1000-100.csv", "trainR-1000-100.csv", "test-1000-100.csv", "testR-1000-100.csv"),
    "crime": ("train-crime.csv", "trainR-crime.csv", "test-crime.csv", "testR-crime.csv"), 
    "wine": ("train-wine.csv", "trainR-wine.csv", "test-wine.csv", "testR-wine.csv") }


# In[12]:


for key,data in data_files.items():
    design_matrix_train = pd.read_csv(data[0], header = None)
    response_variable_train = pd.read_csv(data[1], header = None)
    design_matrix_test = pd.read_csv(data[2], header = None)
    response_variable_test = pd.read_csv(data[3], header = None)
    
    MSE_train, MSE_test = MSE_train_test(design_matrix_train, response_variable_train,design_matrix_test,response_variable_test)
    MSE_vs_lambda_plot(MSE_train, MSE_test, key)
#     print(min(MSE_test))
#     print(MSE_test.index(min(MSE_test)) +1)


# In[ ]:





# In[ ]:





# In[ ]:




