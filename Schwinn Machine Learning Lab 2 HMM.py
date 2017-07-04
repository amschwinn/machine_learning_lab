# -*- coding: utf-8 -*-
"""
Machine Learning Practical Session

Hidden Markov Models

By: Austin Schwinn and Usama Javaid

Mar 6, 2017
"""
#############################
#Exercise 1: 
#Weather HMM Example

import numpy as np
from hmmlearn import hmm
import math

#Create our states
states = ["Rainy", "Sunny"]
n_states = len(states)

#Our observations of what we do based on the state
observations = ["walk", "shop", "clean"]
n_observations = len(observations)

#Initialize our HMM model
#With map parameter
#model = hmm.MultinomialHMM(n_components=n_states, init_params="",
#                           n_iter=10,algorithm='map',tol=0.00001)
#With viterbi
model = hmm.MultinomialHMM(n_components=n_states, init_params="",
                           n_iter=10,algorithm='viterbi',tol=0.00001)
#help(hmm.MultinomialHMM)

#Initial State Occupation Distribution
#(60% raining, 40% sunnt)
model.startprob_ = np.array([0.6, 0.4])

#Matrix of transition probabilities between states
#From each state to another state
#(From 70% from R to R, 30% from R to S. 40% from S to R,70% S to S)
model.transmat_ = np.array([
[0.7, 0.3],
[0.4, 0.6]
])

#Probability of emitting a given symbol when in each state
#(R: 10% walk, 40% shop, 50% clean. S: 60% walk, 30% shop, 10% clean)
model.emissionprob_ = np.array([
[0.1, 0.4, 0.5],
[0.6, 0.3, 0.1]
])

#Creating a feature matrix with 3 samples
gen1 = model.sample(3)
print(gen1)
help(hmm.MultinomialHMM.sample)

#Create feature matrix and state sequence with 5 sampels
seqgen2, stat2= model.sample(5)
print(seqgen2)
print(stat2)

#Another 2 sample feature matrix
gen3=model.sample(2)
print(gen3)

#The logarithmic probability of sequence
sequence1 = np.array([[2, 1, 0, 1]]).T
logproba=model.score(sequence1)
print(logproba)
#help(hmm.MultinomialHMM.score)

#Compute log prob and posteriors
logproba_extend=model.score_samples(sequence1)
print(logproba_extend)
#help(hmm.MultinomialHMM.score_sample)

#Find most likely state sequence corresponding to seq1
p = model.predict(sequence1)
print(p)
#help(hmm.MultinomialHMM.predict)

#Compute log prob and posteriors
p_extend = model.score_samples(sequence1)
print(p_extend)

#estimate model parameters
model.fit(sequence1)

#Log prob and parameters after fitting
p_extend = model.score_samples(sequence1)
print(p_extend)

#New model parameters
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([
[0.7, 0.3],
[0.3, 0.7]])
model.emissionprob_ = np.array([
[0.2, 0.3, 0.5],
[0.7, 0.2, 0.1]
])

#Fit new model
model.fit(seqgen2)

#New log prob and parameters
p_extend = model.score_samples(sequence1)
print(p_extend)

#Example with multiple sequences
sequence3 = np.array([[2, 1, 0, 1]]).T
sequence4 = np.array([[2, 1, 0, 1, 1]]).T
sample = np.concatenate([sequence3, sequence4])
lengths = [len(sequence3), len(sequence4)]
model.fit(sample,lengths)

#State names instead of index
sequence = np.array([[2, 1, 0, 1]]).T
logprob, state_seq = model.decode(sequence, algorithm="viterbi")
print("Observations", ", ".join(map(lambda x: observations[x], sequence)))
print("Associated states:", ", ".join(map(lambda x: states[x], state_seq)))

###################################################
#Exercise 2
#HMM with string as states
 
A = np.array([[0.45,0.35,0.20],[0.10,0.50,0.40],[0.15,0.25,0.60]])
B = np.array([[1.0,0.0],[0.5,0.5],[0.0,1.0]])
Pi = np.array([0.5,0.3,0.2])

#create the states
states = ["a", "b","c"]
n_states = len(states)

#start the model
model = hmm.MultinomialHMM(n_components=n_states, init_params="",
                           n_iter=10,algorithm='map',tol=0.00001)
#Fit the model
model.startprob_=Pi
model.emissionprob_=B
model.transmat_=A

#Compute the probability of the string abbaa
string = np.array([[0,1,1,0,0]]).T
logproba=model.score(string)
#probabiltiy of abbaa
print(logproba)
print(math.exp(logproba))

# Apply Baum Welh with only one iteration and check the probability
# of the string. Do for 1, 15, and 150 iterations
iterations = np.array([1,15,150])

for i in iterations:
    #start the model
    model2 = hmm.MultinomialHMM(n_components=n_states, init_params="",
                           n_iter=i,algorithm='map',tol=0.00001)
    #Fit the model
    model2.startprob_=Pi
    model2.emissionprob_=B
    model2.transmat_=A
    
    model2.fit(string)

    #Probabibility after 1 Baum-Welch iteration
    print(i)
    logproba_extend=model2.score_samples(string)
    print("probability:")
    print(math.exp(logproba_extend[0]))
    print(logproba_extend[0])
    print("parameters:")
    print(logproba_extend[1])
    
## Now create HMM with 5 states with params initialized at any non-neg
A = np.array([[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2]])
B = np.array([[1.0,0.0],[0.5,0.5],[0.0,1.0],[0.5,0.5],[0.0,1.0]])
Pi = np.array([0.2,0.2,0.2,0.2,0.2])

#create the states
states = ["a", "b","c","d","e"]
n_states = len(states)

#start the model
model = hmm.MultinomialHMM(n_components=n_states, init_params="",
                           n_iter=10,algorithm='map',tol=0.00001)
#Fit the model
model.startprob_=Pi
model.emissionprob_=B
model.transmat_=A

#Compute the probability of the string abbaa
string = np.array([[0,1,1,0,0]]).T
logproba=model.score(string)
#probabiltiy of abbaa
print(logproba)
print(math.exp(logproba))


#########################################################
#Exercise 3
#HMM for sequence of string states
A = np.array([[0.4,.6],[.52,.48]])
B = np.array([[0.49,0.51],[0.40,0.60]])
Pi = np.array([0.31,0.69])

states = ["a", "b"]
n_states = len(states)

#start the model
model = hmm.MultinomialHMM(n_components=n_states, init_params="",
                           n_iter=10,algorithm='map',tol=0.00001)
#Fit the model
model.startprob_=Pi
model.emissionprob_=B
model.transmat_=A

#Concatenate into L1
seq11 = np.array([[0,0,0,1,1]]).T
seq12 = np.array([[0,1,0,0,1,1,1]]).T
seq13 = np.array([[0,0,0,1,0,1,1]]).T
seq14 = np.array([[0,0,1,0,1]]).T
seq15 = np.array([[0,1]]).T
l1 = np.concatenate([seq11,seq12,seq13,seq14,seq15])
len_l1 = [len(seq11),len(seq12),len(seq13),len(seq14),len(seq15)]

#fit with the concatenated sequence
model.fit(l1,len_l1)

#output the model
logproba_extend=model.score_samples(l1)
print(logproba_extend[1])

#Concatenate into L2 
seq21 = np.array([[1,1,1,0,0]]).T
seq22 = np.array([[1,0,1,1,0,0]]).T
seq23 = np.array([[1,1,1,0,1,0,0]]).T
seq24 = np.array([[1,1,0,1,1,0]]).T
seq25 = np.array([[1,1,0,0]]).T
l2 = np.concatenate([seq21,seq22,seq23,seq24,seq25])
len_l2 = [len(seq21),len(seq22),len(seq23),len(seq24),len(seq25)]

#fit model with l2
model.fit(l2,len_l2)

#Output the model
logproba_extend=model.score_samples(l2)
print(logproba_extend[1])

#concatenate l3
seq31 = np.array([[0,0,1,0,1,1,1]]).T
seq32 = np.array([[1,1,0,1,0,0,0]]).T
                
print(math.exp(model.score(seq31)))
print(model.score(seq31))
print(math.exp(model.score(seq32)))
print(model.score(seq32))

###################################################
#Exercise 4 
#Handwritten digits classification with HMM

#Load first digit
from sklearn.model_selection import train_test_split
import random
digits1 = []
i = 0
lines = np.loadtxt('digit_strings/1.txt')
for line in lines: 
    digits1.append(','.join(str(int(lines[1]))).split(","))
    digits1[i] = map(int,digits1[i])
    i=i+1
digits1 = np.asarray(digits1)
   
i=0
for i in range(len(digits1)):
    digits1[i] = np.array([i]).T
    i=i+1   
#Split into training and test
train1,test1,=train_test_split(digits1,test_size=0.2,random_state=random.seed())
#check=np.array([train1[1]]).T
#concatenate our multiple sequences into 1
lengths1=[]
for i in train1:
    lengths1.append(len(i))
sample1=np.concatenate(np.array([train1]))

#start the model
n_states=8
model1 = hmm.MultinomialHMM(n_components=n_states, init_params="",
                           n_iter=10,algorithm='map',tol=0.00001)

#Build our intial emission probs
B = np.array([range(len(sample1))]*n_states)
for i in range(n_states):
    for j in range(len(sample1)-2):
        B[i,j] = (random.randint(0,(100/len(sample1)))/100)
    B[i,len(sample1)-1] = 1-sum(B[i,range(len(sample1)-2)])


model1.emissionprob_=B


model1.fit(sample1,lengths1)
