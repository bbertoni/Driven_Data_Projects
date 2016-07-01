# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:42:43 2016

@author: bridget
"""
import numpy as np
from pandas import read_csv
from scipy.optimize import fmin_bfgs
import csv
import random


def scale(X):
    onevec = np.ones(X.shape[1])
    mu = np.mean(X,axis=1)
    mumat = np.outer(mu,onevec)
    dist = np.max(X,axis=1)-np.min(X,axis=1)
    distmat = np.outer(dist,onevec)
    return (X -mumat)/distmat, mu, dist


def rescale(X,mu,dist):
    onevec = np.ones(X.shape[1])
    mumat = np.outer(mu,onevec)
    distmat = np.outer(dist,onevec)
    return (X -mumat)/distmat


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0*z))
    return s
    
    
def compute_cost(theta,X,y,lambd):
    m = len(y)
    z = np.dot(np.transpose(theta),X)
    h = sigmoid(z)
    theta0 = theta
    theta0[0] = 0
    J = ( (1.0/(2.0*m)) * (-np.dot(y,np.log(h)) - np.dot(1-y,np.log(1-h))) + 
        (1.0/(2.0*m)) * lambd * np.dot(theta0,theta0) )
#    print (J)
    return J


def compute_grad(theta,X,y,lambd):
    m = len(y)
    z = np.dot(np.transpose(theta),X)
    h = sigmoid(z)
    theta0 = theta
    theta0[0] = 0
    grad = (1.0/m) * np.dot(X,(h-y)) + (lambd/m) * theta
    return grad
   
   
def bootstrap(data,num):
    traindata.columns = np.array(['index2','lastdon','number','volume','firstdon','madedon'])
# number and volume contain same info  

#avgd = np.zeros(len(traindata.lastdon))
#for i in range(len(traindata.lastdon)):
#    if traindata.lastdon[i]-traindata.firstdon[i] != 0:
#        avgd[i] = traindata.number[i]/(traindata.firstdon[i]-traindata.lastdon[i])
#    else:
#        avgd[i] = traindata.number[i]/(traindata.lastdon[i])

# traindata = traindata[traindata.lastdon<50] # 4

    X0 = np.array([traindata.lastdon,traindata.number])

# 4: X0 = np.array([traindata.lastdon,traindata.lastdon*traindata.number,traindata.number,
#                    traindata.lastdon**2]) 

#X0 = np.array([traindata.lastdon,traindata.number,traindata.firstdon]) 
              
# X = np.array([traindata.lastdon,traindata.number,traindata.firstdon,
#              traindata.lastdon**2,traindata.number**2,traindata.firstdon**2,
#              traindata.lastdon*traindata.number,traindata.lastdon*traindata.firstdon,
#              traindata.number*traindata.firstdon]) 
    lambd = 0
    if num == 1:
        X, mu, dist = scale(X0) # feature scaling and normalization
        X = np.vstack((np.ones(len(X[0])),X)) # add a row of ones
        y = traindata.madedon # "correct" answers to train with
        theta = np.zeros(X.shape[0])
        thetamin = fmin_bfgs(f=compute_cost,x0=theta,fprime=compute_grad,args=(X,y,lambd))
        prob = sigmoid(np.dot(thetamin,X))
        pred = np.zeros(len(prob))
        pred[prob>=0.5] = 1
        print("Percent correct = "+str(sum(pred==y)/len(y)))
        thetamin_avg = thetamin
        seeds = 0
    else:
        m = len(data)
        seeds = np.zeros(num)
        thetavals = np.zeros((num,3))
        for i in range(num):
            X0 = np.array([traindata.lastdon,traindata.number])
            seed = random.randrange(0,10**8,1)
            np.random.seed(seed) # set the seed so this code is reproducible if necessary
            bootvals = np.random.choice(m,m,replace=True) # sample points with replacement
            X0 = X0[:,bootvals]
            if i == 0:
                X, mu, dist = scale(X0) # feature scaling and normalization
            else:
                X = rescale(X0,mu,dist)
            X = np.vstack((np.ones(len(X[0])),X)) # add a row of ones
            y = traindata.madedon[bootvals].values # "correct" answers to train with
            theta = np.zeros(X.shape[0])
            thetamin = fmin_bfgs(f=compute_cost,x0=theta,fprime=compute_grad,args=(X,y,lambd))
            thetavals[i] = thetamin
            seeds[i] = seed
        thetamin_avg = np.mean(thetavals,axis=0)
    return thetamin_avg, mu, dist, seeds
    
    
def jackknife(data,num):
    traindata.columns = np.array(['index2','lastdon','number','volume','firstdon','madedon'])
# number and volume contain same info  
    X0 = np.array([traindata.lastdon,traindata.number])
    lambd = 0
    if num == 1:
        X, mu, dist = scale(X0) # feature scaling and normalization
        X = np.vstack((np.ones(len(X[0])),X)) # add a row of ones
        y = traindata.madedon # "correct" answers to train with
        theta = np.zeros(X.shape[0])
        thetamin = fmin_bfgs(f=compute_cost,x0=theta,fprime=compute_grad,args=(X,y,lambd))
        prob = sigmoid(np.dot(thetamin,X))
        pred = np.zeros(len(prob))
        pred[prob>=0.5] = 1
        print("Percent correct = "+str(sum(pred==y)/len(y)))
        thetamin_avg = thetamin
    else:
        m = len(data)
        thetavals = np.zeros((m,3))
        for i in range(m-1):
            X0 = np.array([traindata.lastdon,traindata.number])
            X0 = np.delete(X0,i,1) # remove one data point
            if i == 0:
                X, mu, dist = scale(X0) # feature scaling and normalization
            else:
                X = rescale(X0,mu,dist)
            X = np.vstack((np.ones(len(X[0])),X)) # add a row of ones
            y = np.delete(traindata.madedon.values,i) # "correct" answers to train with
            theta = np.zeros(X.shape[0])
            thetamin = fmin_bfgs(f=compute_cost,x0=theta,fprime=compute_grad,args=(X,y,lambd))
            thetavals[i] = thetamin
        thetamin_avg = np.mean(thetavals,axis=0)
    return thetamin_avg, mu, dist

###############################################################################################


infile ="training_data.csv"
traindata = read_csv(infile, header = 0)

thetamin, mu, dist = jackknife(traindata,2)

infile ="test_data.csv"
testdata = read_csv(infile, header = 0) 

testdata.columns = np.array(['index2','lastdon','number','volume','firstdon'])      

X2 = np.array([testdata.lastdon,testdata.number]) 

# 4: X2 = np.array([testdata.lastdon,testdata.lastdon*testdata.number,testdata.number,
#                    testdata.lastdon**2])

#X2 = np.array([testdata.lastdon,testdata.number,testdata.firstdon]) 
         
X2 = rescale(X2,mu,dist)
#X2 = scale(X2)[0]
X2 = np.vstack((np.ones(len(testdata.lastdon)),X2))       

prob2 = sigmoid(np.dot(thetamin,X2))

file = open("blood_submission11.csv", "w",newline='')
c = csv.writer(file)
c.writerow(["","Made Donation in March 2007"])

for i in range(len(prob2)):
    c.writerow([testdata.index2[i],prob2[i]])
    
file.close()


# http://blog.yhat.com/posts/logistic-regression-and-python.html    
# http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html