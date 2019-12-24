#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


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



design_matrix_train_1000_100 = pd.read_csv("train-1000-100.csv", header = None)
response_variable_train_1000_100 = pd.read_csv("trainR-1000-100.csv", header = None)
design_matrix_test_1000_100 = pd.read_csv("test-1000-100.csv", header = None)
response_variable_test_1000_100 = pd.read_csv("testR-1000-100.csv", header = None)


MSE_test= list()
lambda_ = (0.5, 27, 149,)
for value in lambda_:
    for x in range(100,1001,100):  
        MSE_test.append(MSE_train_test(design_matrix_train_1000_100[:x], response_variable_train_1000_100[:x], design_matrix_test_1000_100,response_variable_test_1000_100, value))

# Learning curve plotting
        
x1 = range(10,101,10)
y1 = MSE_test[:10]
y2 = MSE_test[10:20]
y3 = MSE_test[20:30]
plt.plot(x1, y1, label = "λ = {}".format(lambda_[0]))
plt.plot(x1, y2, label = "λ = {}".format(lambda_[1]))
plt.plot(x1, y3, label = "λ = {}".format(lambda_[2]))
plt.xlabel('Train set size in percentage')
plt.ylabel("Mean Squared Error")
plt.legend(loc=10)
plt.title("MSE test vs percentage data size for three values of λ")
# plt.yticks(np.arange(min(MSE_test)[0], max(MSE_test)[0] + 1))
plt.xticks(range(0,101,10))
plt.rcParams["figure.figsize"]=(20,10)
plt.show()

print("Min Test MSE for lamda 0.5", min(y1), "Max test MSE for lamda value 0.5", max(y1),
     "Min Test MSE for lamda 27", min(y2), "Max test MSE for lamda value 27", max(y2),
     "Min Test MSE for lamda 149", min(y3), "Max test MSE for lamda value 149", max(y3),)


# In[ ]:




