#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# In[2]:


def MSE_train_test(design_matrix_train, response_variable_train, design_matrix_test, response_variable_test,value):
    
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




def MSE_and_lambda_and_final_MSE(design_matrix_train, response_variable_train, design_matrix_test, response_variable_test):
# MSE calculation using stratified cross validation

    end = int(len(design_matrix_train))
#     print("end", end)
    hop = int(end/10)
#     print("hop", hop)
    MSE_test= list()
    MSE_average = list()
    for value in range(0,150): # values on lambda
        for i in range(0, 10*hop, hop):
                b=i + hop
    #             print(i,b)
                test_design = design_matrix_train[i:b]
                test_response = response_variable_train[i:b]
                train_design = np.concatenate((design_matrix_train[0:i],design_matrix_train[b:]), axis=0)
                train_response = np.concatenate((response_variable_train[0:i],response_variable_train[b:]), axis=0)
                MSE_test.append(MSE_train_test(train_design, train_response, test_design, test_response, value))

    # finding average MSE value for each cross validation set for each value of lambda
#     print(MSE_test)
    avg_MSE = list()
    end1 = int(len(MSE_test))
    for a in range(0,end1,10):
        add = 0
        for x in MSE_test[a:a+10]:
            add += x[0]
        avg_MSE.append(add/len(MSE_test[a:a+10]))

    # choosing the value of lambda with lowest MSE average
#     print("length avg MSE", len(avg_MSE))
#     print("min MSE",min(avg_MSE))
    lambda_value = avg_MSE.index(min(avg_MSE))

    # finding MSE on test data using lambda value found using stratfied cross validation

    MSE_stratified = list()
    MSE_stratified.append(MSE_train_test(design_matrix_train, response_variable_train, design_matrix_test, response_variable_test, lambda_value))
    
    return MSE_stratified, lambda_value  # run time is also needed
            

    


# In[6]:


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
    
    MSE_stratified, lambda_value = MSE_and_lambda_and_final_MSE(design_matrix_train, response_variable_train,design_matrix_test,response_variable_test)
    output[key]= {"Test MSE of {}".format(key):round(MSE_stratified[0][0],4), 
                  "Lambda value of {}". format(key): lambda_value,
                  "Run time of {}".format(key): round(time.time()- start_time,3)}


# In[7]:


print(output)


# In[ ]:





# In[ ]:





# In[ ]:




