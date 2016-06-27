# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:42:43 2016

@author: bridget
"""
import numpy as np
from pandas import read_csv
from scipy.optimize import fmin_bfgs
import csv


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
    print (J)
    return J

def compute_grad(theta,X,y,lambd):
    m = len(y)
    z = np.dot(np.transpose(theta),X)
    h = sigmoid(z)
    theta0 = theta
    theta0[0] = 0
    grad = (1.0/m) * np.dot(X,(h-y)) + (lambd/m) * theta
    return grad

infile ="training_data.csv"
traindata = read_csv(infile, header = 0)

traindata.columns = np.array(['index2','lastdon','number','volume','firstdon','madedon'])
# number and volume contain same info  

#avgd = np.zeros(len(traindata.lastdon))
#for i in range(len(traindata.lastdon)):
#    if traindata.lastdon[i]-traindata.firstdon[i] != 0:
#        avgd[i] = traindata.number[i]/(traindata.firstdon[i]-traindata.lastdon[i])
#    else:
#        avgd[i] = traindata.number[i]/(traindata.lastdon[i])

X0 = np.array([traindata.lastdon,traindata.number]) 

#X0 = np.array([traindata.lastdon,traindata.number,traindata.firstdon]) 
              
# X = np.array([traindata.lastdon,traindata.number,traindata.firstdon,
#              traindata.lastdon**2,traindata.number**2,traindata.firstdon**2,
#              traindata.lastdon*traindata.number,traindata.lastdon*traindata.firstdon,
#              traindata.number*traindata.firstdon]) 
X = scale(X0)[0]
X = np.vstack((np.ones(len(traindata.lastdon)),X))

y = traindata.madedon

theta = np.zeros(X.shape[0])
lambd = 0
thetamin = fmin_bfgs(f=compute_cost,x0=theta,fprime=compute_grad,args=(X,y,lambd))

prob = sigmoid(np.dot(thetamin,X))

pred = np.zeros(len(prob))
pred[prob>=0.5] = 1

print("Percent correct = "+str(sum(pred==y)/len(y)))   

infile ="test_data.csv"
testdata = read_csv(infile, header = 0) 

testdata.columns = np.array(['index2','lastdon','number','volume','firstdon'])      

X2 = np.array([testdata.lastdon,testdata.number]) 

#X2 = np.array([testdata.lastdon,testdata.number,testdata.firstdon]) 
         
X2 = rescale(X2,scale(X0)[1],scale(X0)[2])
#X2 = scale(X2)[0]
X2 = np.vstack((np.ones(len(testdata.lastdon)),X2))       

prob2 = sigmoid(np.dot(thetamin,X2))

file = open("blood_submission4.csv", "w",newline='')
c = csv.writer(file)
c.writerow(["","Made Donation in March 2007"])

for i in range(len(prob2)):
    c.writerow([testdata.index2[i],prob2[i]])
    
file.close()


# http://blog.yhat.com/posts/logistic-regression-and-python.html    
# http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html