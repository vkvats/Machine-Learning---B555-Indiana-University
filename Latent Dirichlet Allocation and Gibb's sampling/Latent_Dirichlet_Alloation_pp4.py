#!/usr/bin/env python
# coding: utf-8

import random, re
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import csv
from scipy.stats import logistic as logit
import time
import matplotlib.pyplot as plt


# # LDA implementation

class LatentDirichletAllocation:
    
    def __init__(self, data, n_docs, n_topics, alpha, beta, n_iters ):
        self.documents = data # data in dictionar form that is read from directory
        self.n_docs = n_docs # number of documents in the directory
        self.n_topics = n_topics # number of given topics
        self.alpha = alpha # topic sampling prior
        self.beta = beta # word sampling prior
        self.n_iters = n_iters # given number of iterations
        self.document_indices = np.array([],dtype = int) #  document indices d(n)
        self.topic_indices = np.array([],dtype = int) #  topic indices z(n)
        self.ordered_data = np.array([],dtype = int) # data is ordered as per document name from 1 to D
        self.n_words = 0 # total number of words in corpus
        self.vocab = None # vocabulary of unique words
        self.len_vocab = 0 # length of vocabulary of unique words.
        self.probability = np.zeros(self.n_topics) # probability distribution 
        self.word_indices = None # indices of words in the corpus.
        self.permutation = None # random permutation of words in corpus.
        self.CD = None # initialise topic counts per document C_d : n_docs x topics
        self.CT = None # initialise word count per topic C_t : topics x vocab_count
        self.top_five = None # top_five value after Gibb's sampling for each topic.
        
    # This method orders the data as per the sequential documents provided in the directory
    # after ordering the data sequentially it assigns topic indices to all words per document.
    # It also tags every words with the documents number sequentially, called as document indices.
    def ordered_data_and_indices(self):
        for i in range(1, self.n_docs+1):
            # collecting all words into one big array in ordered documents
            length = len(self.documents[str(i)])
            self.ordered_data = np.append(self.ordered_data,self.documents[str(i)] )
            y = np.random.randint(1, self.n_topics+1 , [1, length])
            self.topic_indices = np.append(self.topic_indices, y)
            self.documents[str(i)]= self.documents[str(i)],y          
            x = i*np.ones([1,length],dtype = int)
            self.document_indices = np.append(self.document_indices, x)

    # This methods creaste the word indices for all the words in the corpus.   
    # for doing that, we need to have a unique set of vocabulary and an index value assigned to it.
    # that index values is used to generate the word indices for the whole corpus.
    def create_word_indices(self):
        self.n_words = len(self.ordered_data) # number of total number of words in corpus combinint all documents
        self.vocab = list(set(self.ordered_data)) # set of all unique words.
        self.len_vocab = len(self.vocab)
        word_ind = defaultdict.fromkeys(self.vocab,0)
        for i in range(self.len_vocab):
            word_ind[self.vocab[i]] = i+1
        self.word_indices = np.zeros(self.n_words, dtype = int)
        for i in range(self.n_words):
            self.word_indices[i] = word_ind[self.ordered_data[i]]
            
    # The data randomisation part is done by this method and then it counts the value of CD and CT     
    def randomise_and_counts(self):
        self.permutation = np.random.permutation(range(0,self.n_words))
        self.CD = np.zeros((self.n_docs, self.n_topics))
        self.CT = np.zeros((self.n_topics, self.len_vocab))
        for i in range(self.n_docs):
            for j in range(self.n_topics):
                self.CD[i,j] = np.sum(self.documents[str(i+1)][1]== j+1)
        for i in range(self.n_topics):
            for j in range(self.len_vocab):
                self.CT[i,j] = sum((self.topic_indices == i+1) * (self.word_indices == j+1))
                
    # This method implements the collapsed gibbs sampling algorithm step by step as given in the 
    # assignment 4. the only change is that, I have used all indecies starting from 1 so i need to
    # subtract 1 while checking for index in CD and CT everytime.
    def collapsed_gibbs_sampling(self):
        for i in range(self.n_iters):
            for n in range(0,self.n_words):
                word_n = self.permutation[n]
                word = self.word_indices[word_n]
                topic = self.topic_indices[word_n]
                doc = self.document_indices[word_n]
                self.CD[doc-1,topic-1] = self.CD[doc-1,topic-1]- 1
                self.CT[topic-1,word-1] = self.CT[topic-1,word-1]- 1
                for j in range(0,self.n_topics):
                    self.probability[j] = (self.CT[j,word-1] + self.beta)/(self.len_vocab*self.beta + np.sum(self.CT[j,:]))*(self.CD[doc-1,j] + self.alpha)/(self.n_topics*self.alpha + np.sum(self.CD[doc-1,:]))
                self.probability = self.probability/np.sum(self.probability)
                topic = np.random.choice(range(1, self.n_topics+1), p = self.probability)
                self.topic_indices[word_n] = topic
                self.CD[doc-1,topic-1] = self.CD[doc-1,topic-1] + 1
                self.CT[topic-1,word-1] = self.CT[topic-1,word-1] + 1
                
    # This method finds out the top 'N' most frequent words,
    # right now the current value is set to N =5 as per assignment.
    # this function returns the top five values.
    def top_frequent(self, n):
        self.top_five = np.zeros([self.n_topics,n],dtype = "object") # need to define data type as not providing it is giving error.
        for i in range(self.n_topics):
            t= np.argsort(self.CT[i])
            t = t[-n:][::-1]
            for j in range(n):
                self.top_five[i,j] = self.vocab[t[j]]
        return self.top_five


# # Logictic regression implementation


# this whole section of Logistic regression has been taken directly from Assignment 3 implementation
# without any change in any part of implementation in GLM class.

class GLM: # the Generalised Linear Model class
    
    ###### common operation for all models ########  
    
    def __init__(self, phi, labels, alpha_value, distribution_name, test_label, test_data):
        self.phi_matrix = np.insert(np.array(phi), 0, 1, axis=1) # added 1 at first column of phi for w0.
        self.label = np.array(labels) # the value R in second derivative.
        self.alpha = alpha_value
        self.method = distribution_name.lower()
        self.w_0 = np.zeros((np.shape(self.phi_matrix)[1],1))
        self.label_test = test_label
        self.phi_test = np.insert(np.array(test_data), 0, 1, axis=1)
        
    # this function will find the first derivating of the model and reutrn first derivative.
    def first_derivative(self, di: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.subtract(self.phi_matrix.transpose().dot(di), self.alpha*w)
     
    #this function will find the second derivative of model and return inverse of hassian.
    def second_derivative(self, ri: np.ndarray) -> np.ndarray:
        phit_R_phi = self.phi_matrix.transpose().dot(ri).dot(self.phi_matrix)
        hessian = np.add(phit_R_phi, self.alpha*(np.identity(len(phit_R_phi))))
        return np.linalg.inv(hessian)
     
    # this function will implement the newton-raphson method and return Wmap, 
    #number of iteration and convergence time.
    def newton_raphson(self, s=1) -> np.ndarray:
        start_time = time.time()
        w = self.w_0
        # N_R loop for convergance
        for itr in range(1,101): # for 100 iteration as specified in PP3
            w_old = w
            ri, di = self.yi_di_ri(w, s)
            derivative = self.first_derivative(di,w)
            hessian_inv = self.second_derivative(ri)
            w = w + hessian_inv.dot(derivative)
            if itr >1:
                mat = np.subtract(w, w_old)
                if np.linalg.norm(mat)/ np.linalg.norm(w_old) < 0.001:
                    end_time = round(time.time()- start_time,4)
                    return w, itr, end_time 
                elif itr == 100:
                    end_time = round(time.time()- start_time,4)
                    return w, itr, end_time            
          
        ############ Model based operation ############
        
    def yi_di_ri(self, w: np.ndarray, s=1) -> np.ndarray:
        # a_i = phi_tanspose.parameter_W
        ai = self.phi_matrix.dot(w).reshape(len(self.phi_matrix),1)
        ########## Logistic Regression #############
        # it will return the ri and di value for logistic regression.
        if self.method == "logistic":
            # yi = sigmoid function
            yi = 1/(1+ np.exp(-1*ai))
            di = np.subtract(self.label, yi) # (t-y)
            # R diag(yi(1- yi))
            ri = np.diagflat(yi*np.subtract(np.ones(np.shape(yi)), yi))
            return ri, di

              
    def predict(self, wmap):
        ###### Logistic Regression ######
        # prediction and error calculation for logic regression.
        # return average error.
        if self.method == "logistic":           
            # predicting the value
            ai_z = self.phi_test.dot(wmap).reshape(len(self.phi_test),1)
            pred = []
            for x in np.nditer(ai_z):
                if x >= 0:
                    pred.append(1)
                elif x < 0:
                    pred.append(0)
            # counting acuracy and error (mismatch for logistic regression)
            i = 0  
            accuracy_sum = 0
            for y in np.nditer(self.label_test):
                if y != pred[i]:
                    i +=1
                else:
                    accuracy_sum +=1
                    i +=1
            return accuracy_sum/len(pred)


# # Supporting function for both part.

# this function randomises the data set and then split it into examples and labels and returns the value
def randomise_and_split(complete_data):
    # random shuffling the whole data along with labels.
    np.random.shuffle(complete_data)
    # dividing data into test data and train data
    division_point = int(len(complete_data)/3)
    test_matrix = complete_data[:division_point]
    train_matrix = complete_data[division_point :]  
    # separating examples and labels.
    test_label = test_matrix[:,[-1]]
    test_data = np.delete(test_matrix, -1, axis=1)   
    train_label = train_matrix[:,[-1]]
    train_data = np.delete(train_matrix, -1, axis=1) 
    return test_label, test_data, train_label, train_data

# this function calculates the model statics values like mean value and standard deviation values
# it returns mean and standard deviation values
def model_statics_calculation(mean_values):
    mean_fold=[]
    std_fold = []
    for j in range(0,10):
        err = []
        sd = []
        for i in range(0,30):
            err.append(mean_values[10*i + j])
        mean_fold.append(sum(err)/len(err))
        std_fold.append(np.std(err))
    return mean_fold, std_fold

# this function plots the learning curve as per requirement.
def learning_curve(error_lda, error_bag, sd_lda, sd_bag):
    x1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.figure()
    plt.errorbar(x1, error_lda, sd_lda, label = "mean accuracy LDA")
    plt.errorbar(x1, error_bag, sd_lda, label = "mean accuracy bag of words")
    plt.xlabel('Sample Size')
    plt.ylabel("Mean Accuracy")
    plt.legend(loc="center")
    plt.title("Learning curve comparison for LDA model and bag-of-words modal")      
    plt.rcParams["figure.figsize"]=(15,7)
    return plt.show()

# this function reads all file and store it in dictionary form.
# it returns the dictionary.
def read_all_files(directory):
    documents = {}
    doc_len = {}
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            with open(directory + filename, "r") as f:
                text = f.read()
                doc = re.findall(r"[\w']+", text)
                doc_len[filename] = len(doc)
                documents[filename]= doc
                f.close()
    return documents


# # Main program

# please provide any other directory name if the directory names are not same as given here. 
# directory = "pp4data/artificial/" 
directory = "pp4data/20newsgroups/"


# reading the original labels from the directory.
# please check the directory when the program is not implemented accordingly.
labels = np.array(pd.read_csv("pp4data/20newsgroups/index.csv", header = None)[1]).reshape(200,1)

# prior values and other constants
docs = 200 # number of documents D
topics = 20 # number of topics K
alpha = 5.0/topics 
beta = 0.01
iterations = 500


# load all the documents in code.
documents = read_all_files(directory)
# I am recording the implementation time from here. it will be printed at the end of this part of execution.
start_time = time.time()
# creating a lda object.
lda = LatentDirichletAllocation(documents, docs, topics, alpha, beta, iterations )
# class based method operations.
lda.ordered_data_and_indices()
lda.create_word_indices()
lda.randomise_and_counts()
# gibbs sampling implementation.
lda.collapsed_gibbs_sampling()
# choosing "N" most frequent wrods after gibbs sampling conlcudes.
# provide a value of n: top most n frequent words for each topic.
most_frequent= lda.top_frequent(n=5) 
end_time = time.time()
print(end_time-start_time, "secs for Gibbs sampling.") 


# writing the top five most frequent words for each topic into a CSV file.
# pd.DataFrame(most_frequent).to_csv("topicwords.csv", header = False, index = False)


# # Part 2 : Classification

# Calculating features for both type of implementation LDA and Bag of words model for 
# comparing it in Logistic regression output.

bag_of_words_features = np.zeros([lda.n_docs,lda.len_vocab])
for i in range(lda.n_docs):
    j=0
    for word in lda.vocab:
        bag_of_words_features[i,j] = lda.documents[str(i+1)][0].count(word)/len(lda.documents[str(i+1)][0])
        j +=1
        
lda_topic_representation = np.zeros([lda.n_docs,lda.n_topics])
for i in range(lda.n_docs):
    for j in range(lda.n_topics):
        lda_topic_representation[i,j] = (lda.CD[i,j] + lda.alpha)/(lda.n_topics*lda.alpha + np.sum(lda.CD[i,:]))





# adding the true labels to the features for future randomaization and cross validation split.
data_and_labels_bag_model = np.concatenate((bag_of_words_features, labels), axis=1)
data_and_labels_lda = np.concatenate((lda_topic_representation, labels), axis=1)



###### prediction and calculation part for LDA model ########

# running LDA model features in Logistic regression and noting the error for the same.
run_time_lda = []
iterations_lda = []
errors_lda = []
accuracy_lda =[]
# repeating the process 30 times for LDA model features.
for i in range(30):    
    # Call randomise_and_split to get randomised splitted data.
    test_label_l, test_data_l, train_label_l, train_data_l = randomise_and_split(data_and_labels_lda)

    # divide this data into 10 subsets 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.
    step = int(len(train_data_l)/10)+1
    for x in range(step, len(train_data_l)+step, step ):
        logistic = GLM(train_data_l[:x], train_label_l[:x], 0.01, "logistic", test_label_l, test_data_l )
        # getting W_map, number of iteration and convergence time 
        w_map_l, iterr_l, running_time_l = logistic.newton_raphson()
        run_time_lda.append(running_time_l)
        iterations_lda.append(iterr_l)
        # Do prediction using W_map and note the total error for each iteration.
        accuracy = logistic.predict(w_map_l)
        accuracy_lda.append(accuracy)
  

###### prediction and calculation part bag of words model ########

# running bag of words model features in Logistic regression and noting the error for the same.
run_time_bag = []
iterations_bag = []
errors_bag = []
accuracy_bag = []

# repeating the process 30 times for bag of word model features.
for i in range(30):    
    # Call randomise_and_split to get randomised splitted data.
    test_label_, test_data_, train_label_, train_data_ = randomise_and_split(data_and_labels_bag_model)

    # divide this data into 10 subsets 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.
    step = int(len(train_data_)/10)+1
    for x in range(step, len(train_data_)+step, step ):
        logistic = GLM(train_data_[:x], train_label_[:x], 0.01, "logistic", test_label_, test_data_ )
        # getting W_map, number of iteration and convergence time 
        w_map_, iterr_, running_time_ = logistic.newton_raphson()
        run_time_bag.append(running_time_)
        iterations_bag.append(iterr_)
        # Do prediction using W_map and note the total error for each iteration.
        accuracy = logistic.predict(w_map_)
        accuracy_bag.append(accuracy)


# calculating model statistics for both LDA and bag of word models
mean_acc_fold_lda, std_acc_lda = model_statics_calculation(accuracy_lda)
mean_acc_fold_bag, std_acc_bag = model_statics_calculation(accuracy_bag)


# the learning curve plot for both the model.
learning_curve(mean_acc_fold_lda, mean_acc_fold_bag, std_acc_lda, std_acc_bag)




