
# coding: utf-8

# In[1]:

import numpy as np    
import matplotlib.pylab as plt
import csv
import datetime
import sklearn.linear_model as sklin
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
import statsmodels.api as sm
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.svm import SVC 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import ExtraTreesClassifier as ETC
import h5py
from sklearn.feature_selection import VarianceThreshold as vt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import RandomizedPCA
from scipy.stats import mode

import getpass

if getpass.getuser() == 'Pragnya':
	direc = 'C:/Users/Pragnya/Documents/GitHub/LIS/Project4/'
else:
	direc = ''

####### read validation data
Xval=[]
Xtempval=[]
valstep=0
with open(direc+'validate.csv','r') as fin:
    reader = csv.reader(fin,delimiter=',')
    for row in reader:
        for ind in range(0,len(row)):
            Xtempval.append(float(row[ind]))
        Xval.append(Xtempval)
        Xtempval = []
X_val=np.array(Xval)
#X_val_norm=preprocessing.scale(X_val)
X_val_norm = X_val
Y_res=[]


# In[2]:

####### read train data and labels
Y=[]
X=[]
Xtemp=[]
step=0
maxim = 7000
with open(direc+'train.csv','r') as fin:
    reader = csv.reader(fin,delimiter=',')
    for row in reader:
        #print('step: ',step)
        step = step+1
        if step == maxim:
            break
        for ind in range(len(row)):
            Xtemp.append(float(row[ind]))
        X.append(Xtemp)
        Xtemp=[]
        
print('done reading Xtrain')
step = 0
with open(direc+'train_y.csv') as fin:
    reader = csv.reader(fin,delimiter=',')
    for row in reader:
        #print('step: ',step)
        step = step+1
        if step == maxim:
            break
        Y.append(float(row[0]))
    #Y=np.atleast_2d(Y)

print('done reading Ytrain')
Xtrain=np.array(X)
Ytrain=np.array(Y)


# In[3]:

Ytrain = np.transpose(Ytrain)


# In[4]:

print(Ytrain.shape)
print(Xtrain.shape)


# In[5]:

####### define score functions
import math

def score(truth,pred):
    su = 0
    labeled = 0
    for i in range(truth.shape[0]):
        if(truth[i]!=-1):
            print('true label is: ',truth[i])
            print('predicted probability is: ',pred[i,truth[i]])
            su = su - math.log(max(0.0001,pred[i,truth[i]]))
            labeled = labeled+1
    if labeled == 0:
        print('PROBLEM!! DIV BY 0')
    return su/labeled
    #dif = -log(max(0.0001,pred[truth]))
    #return np.divide(sum(dif))


# In[6]:

#X_norm=preprocessing.scale(Xtrain)
X_norm = Xtrain
X_train, X_test, Y_train, Y_test = skcv.train_test_split(X_norm,Ytrain,test_size=0.4, random_state=0)


# In[7]:

#count number of missing labels
noLabels = [1 for i in Y_train if i==-1]
labels = [1 for i in Y_train if not i==-1]
print('unlabeled: ',np.sum(noLabels))
print('labeled: ',np.sum(labels))


# In[8]:

from sklearn.semi_supervised import LabelPropagation as LP
from sklearn.semi_supervised import LabelSpreading as LS


# In[17]:

lspr = LP(gamma = 70)
lspr.fit(X_norm,Ytrain)


# In[15]:

print('nofClasses: ',lspr.classes_)


# In[16]:

pred = lspr.predict(X_norm)
notN = [1 for i in pred if i>0.0]
print(sum(notN))


# In[12]:

Y_pred = lspr.predict_proba(X_test)


# In[13]:

print(Y_pred.shape)


# In[ ]:



