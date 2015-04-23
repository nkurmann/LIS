
"<A>,<B>,<C>,<D>,<E>,<F>,<G>,<H>,<I>,<K+>,<L+>"
"where the fields <A>–<I> contain numbers representing the parameters about" 
"the geometrical and texture-related features. The field <K+> consists of four"
"binary features representing a four-valued categorical feature in one-of-k format 1." 
"Similarly, the field <L+> consists of 40 binary columns in one-of-k format."


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
