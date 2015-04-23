
# coding: utf-8



import numpy as np    
import matplotlib.pylab as plt
import csv
from sklearn import preprocessing
import statsmodels.api as sm
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv

def get_features(A,B,C,D,E,F,G,H,I,K,L):
	return [K,L,A,B,C,D,E,F,G,H,I]
    #return [A]

def read_data_x(inpath):
    X=[]
    with open(inpath,'r') as fin:
        reader=csv.reader(fin,delimiter=',')
        for row in reader:
            A=float(row[0])
            B=float(row[1])
            C=float(row[2])
            D=float(row[3])
            E=float(row[4])
            F=float(row[5])
            G=float(row[6])
            H=float(row[7])
            I=float(row[8])

            k=row[9:13]
            kk=[i for i in range(len(k)) if not k[i]=='0']
            K=sum(kk)
            l=row[13:53]
            ll=[i for i in range(len(l)) if not l[i]=='0']
            L=sum(ll)
            X.append(get_features(A,B,C,D,E,F,G,H,I,K,L))
    X=np.atleast_2d(X)
    #!make K and L categorical
    #X=sm.tools.categorical(X,col=0,drop=False)
    #X=sm.tools.categorical(X,col=1,drop=False)
    return X

def read_data_y(inpath):
    Y=[]
    with open(inpath,'r') as fin:
        reader=csv.reader(fin,delimiter=',')
        for row in reader:
            y=int(row[0])
            z=int(row[1])
            Y.append([y,z])
        Y=np.atleast_2d(Y)
    return Y

def score(gtruth,pred):
    dif = gtruth-pred
    vec_1=[1.0 for x in dif[:,0] if not x==0]
    vec_2=[1.0 for x in dif[:,1] if not x==0]
    scprod=np.sum(vec_1)+np.sum(vec_2)
    return scprod/(2*len(vec_1))


#read data X:
folder = 'C:/Users/Pragnya/Documents/Studium/P2/' #folder='../'
X = read_data_x(folder+'train.csv')
#scale data X (normalize s.t. mean=0, distr.=gaussian, variance=1)
#print(X[0:10,0:2])

X_sc = preprocessing.scale(X[:,2:len(X)])

X_sc2 = np.concatenate((X[:,0:2]),axis=1)
   
#read data y
Y=read_data_y(folder+'train_y.csv')

#scorefun = skmet.make_scorer(score)

#training / test sets:
#X_train, X_test, Y_train, Y_test = skcv.train_test_split(X_sc,Y,test_size=0.4, random_state=0)


#prediction with KNN
from sklearn.neighbors import KNeighborsClassifier


#evtl. semi-supervised learning?




