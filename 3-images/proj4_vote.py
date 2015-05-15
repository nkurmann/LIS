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
####### read validation data

fval=h5py.File('project_data/validate.h5',"r")
data=fval["data"]
Xval=[]
Xtempval=[]
valstep=0
for row in data:
    print ('val',valstep)
    valstep=valstep+1
    for ind in range(len(row)):
	Xtempval.append([float(row[ind])])    
    Xval.append(Xtempval)
    Xtempval=[] 
X_val=np.array(Xval)[:,:,0]
X_val_norm=preprocessing.scale(X_val)


Y_res=[]

for run in range(9):
	print("Starting run %d"%run)
####### read train data and labels
	
	f=h5py.File('project_data/train.h5',"r")
	data=f["data"]
	label=f["label"]
	Y=[]
	X=[]
	Xtemp=[]
	step=0
	for row in data[run*3500:run*3500+12000,:]:
		if step/100 == 0:
			print('row=',run,step)
	
		step=step+1
		for ind in range(len(row)): 
			Xtemp.append([float(row[ind])])    
		X.append(Xtemp)
		Xtemp=[] 
	for row in label[run*3500:run*3500+12000,:]:
		Y.append([float(row[0])])     

	Xtrain=np.array(X)[:,:,0]
	Ytrain=np.array(Y)[:,0]

	X_norm=preprocessing.scale(Xtrain)
	X_train, X_test, Y_train, Y_test = skcv.train_test_split(X_norm,Ytrain,test_size=0.005, random_state=0)

####### define score functions

	def score(gtruth,pred):
	    dif = gtruth-pred
	    vec=[1.0 for x in dif if not x==0]
	    print('y',np.sum(vec))
	    scprod=np.sum(vec)
	    return np.divide(scprod,len(pred))
	   
####### find best estimators

	def get_best_estimator(X, Y ):
	    #best = RFC(n_estimators=1000)
	    #best = KNC(n_neighbors=1000)
	    #best = ETC(n_estimators=1000)
	    best=SVC(kernel='rbf')
	    best.fit(X,Y)
	    #print(best)
	    return best

	n_components=200
	print("Extracting the top %d eigenimages from %d images"%(n_components,X_train.shape[0]))

	pca=RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
	X_train_pca=pca.transform(X_train)
	X_test_pca=pca.transform(X_test)
	X_val_pca=pca.transform(X_val_norm)

	best=get_best_estimator(X_train_pca,Y_train)
	best.fit(X_train_pca, Y_train)

	Y_test_pred=best.predict(X_test_pca)
	scr=score(Y_test,Y_test_pred)
	print('scr:',scr)

	Yval = best.predict(X_val_pca)
        Y_res.append(Yval)
        
y_end=[]
yres=np.array(Y_res)
for nr in range(yres.shape[1]):
    y_end.append(int(mode(yres[:,nr])[0][0]))


	
with open('val_res.csv','w') as fp:
    writer=csv.writer(fp,delimiter=',')
    for row in range(len(y_end)):
	writer.writerow([int(y_end[row])])
 

print("All done!")
		
