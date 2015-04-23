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


def get_features(A,B,C,D,E,F,G,H,I,K,L):
	return [K,L,A,B,C,D,E,F,G,H,I]

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
        #categorical variables
      #  X = np.atleast_2d(X)
      #  X = sm.tools.categorical(X,col=0,drop=False) 
      #  X = sm.tools.categorical(X,col=1,drop=False) 
    return X
def read_data_y(inpath):
    Y=[]                         
    with open(inpath,'r') as fin:
        reader=csv.reader(fin,delimiter=',')
        for row in reader:
            y=int(row[0])
            z=int(row[1])
            Y.append([y,z])
        Y = np.atleast_2d(Y)
    return Y

def score(gtruth,pred):
    dif = gtruth-pred
    vec_1=[1.0 for x in dif[:,0] if not x==0]
    vec_2=[1.0 for x in dif[:,1] if not x==0]
    scprod=np.sum(vec_1)+np.sum(vec_2)
    return scprod/(2*len(vec_1))


"READ DATA"
X=read_data_x('../Projekt_2/train.csv')
Xval=read_data_x('../Projekt_2/validate.csv')
Xtest=read_data_x('../Projekt_2/test.csv')
X_all = np.concatenate((np.array(X),np.array(Xval),np.array(Xtest)),axis=0)

X_all = np.atleast_2d(X_all)
X_all = sm.tools.categorical(X_all,col=0,drop=False) 
X_all = sm.tools.categorical(X_all,col=1,drop=False) 


X_norm = preprocessing.scale(X_all[:,2:len(X_all)])
X_norm = np.concatenate((np.array(X_all[:,0:2]),np.array(X_norm)),axis=1)

X=X_norm[0:len(X),:]
Xval=X_norm[len(X):len(X)+len(Xval),:]
Xtest=X_norm[-len(Xtest):len(X_norm),:]

Y=read_data_y('../Projekt_2/train_y.csv')
scorefun = skmet.make_scorer(score)
#classifier=sklin.RidgeClassifier()


#n_samples = len(X)
#x = np.array(X).reshape((n_samples, -1))

##idea: to find dependencies, we predict one column of y and predict the second taking that in x


X_train, X_test, Y_train, Y_test = skcv.train_test_split(X,Y,test_size=0.2, random_state=0)
#param_grid2 = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4]},{'kernel': ['linear']}]
param_grid1 = {'alpha': np.linspace(1,1000,100)}
#the neg score for only one column
def neg_scorefun(estimator,val,pred):
    gtruth=estimator.predict(val)
    dif=gtruth-pred
    vec=[1.0 for x in dif if not x==0]
    sc=np.sum(vec)/(2*np.array(vec).size)
    return -sc

"BEST ESTIMATORS"
'''def get_best_estimator(X, Y ):
   # grid_search = skgs.GridSearchCV(SVC(C=1), param_grid2, cv=5, scoring=neg_scorefun)
    grid_search = skgs.GridSearchCV(classifier, param_grid1, cv=5, scoring=neg_scorefun)
    #print(grid_search)
    grid_search.fit(X, Y)
    best = grid_search.best_estimator_
    return best
'''
def get_best_estimator(X, Y ):
    best = SVC()
    best.fit(X,Y)
    return best

X_fit_1_train=X_train
Y_fit_1_train=Y_train[:,0]
##Y_fit_1_train=Y_train[:,1]

best_1=get_best_estimator(X_fit_1_train, Y_fit_1_train)
#print(best_1)
#best_1.fit(X_fit_1_train,Y_fit_1_train)

X_fit_2_train=np.concatenate((np.array(X_train),np.transpose(np.array([Y_train[:,0]]))),axis=1)
##X_fit_2_train=np.concatenate((np.array(X_train),np.transpose(np.array([Y_train[:,1]]))),axis=1)
Y_fit_2_train=Y_train[:,1]
##Y_fit_2_train=Y_train[:,0]

best_2=get_best_estimator(X_fit_2_train, Y_fit_2_train)
#print(best_2)
#best_2.fit(X_fit_2_train,Y_fit_2_train)

Y_fit_1_pred=best_1.predict(X_test)
X_test_con=np.concatenate((X_test,np.transpose(np.array([Y_fit_1_pred]))),axis=1)
Y_fit_2_pred=best_2.predict(X_test_con)
Y_pred=np.concatenate((np.transpose(np.array([Y_fit_1_pred])),np.transpose(np.array([Y_fit_2_pred]))),axis=1)
##Y_pred=np.concatenate((np.transpose(np.array([Y_fit_2_pred])),np.transpose(np.array([Y_fit_1_pred]))),axis=1)
scr=score(Y_test,Y_pred)
print('scr:',scr)


"EVALUATE"

Ypred1 = best_1.predict(Xval)
yp1=[int(x) for x in Ypred1]
Xval2=np.concatenate((Xval,np.transpose(np.array([Ypred1]))),axis=1)
Ypred2 = best_2.predict(Xval2)
yp2=[int(x) for x in Ypred2]
Ypred=np.concatenate((np.transpose(np.array([yp1])),np.transpose(np.array([yp2]))),axis=1)
##Ypred=np.concatenate((np.transpose(np.array([yp2])),np.transpose(np.array([yp1]))),axis=1)
with open('../Projekt_2/result_validate.csv','w') as fp:
    writer=csv.writer(fp,delimiter=',')
    for row in range(len(yp1)):
        writer.writerow(Ypred[row,:])
      
    

