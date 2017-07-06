# -*- coding: utf-8 -*-
"""
Machine Learning Practical Session

Support Vector Machines

By: Austin Schwinn and Usama Javaid

Mar 8, 2017
"""
##############################
#Exercise 1: 
#SVM Example

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#Create 4 example
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]) 
y = np.array([-1, -1, 1, 1])

#Create a SVM with default parameters
classif=SVC()

#learn the model according to given data
classif.fit(X,y)

#prediction on a new sample
res=classif.predict([[-0.8, -1]]) 

#Create a mesh to plot in
# grid step
h = .02
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

#the grid is created, the intersections are in xx and yy
mysvc= SVC(kernel='linear', C = 2.0)
mysvc.fit(X,y)

#Predict all the grid
Z2d = mysvc.predict(np.c_[xx.ravel(),yy.ravel()])
Z2d=Z2d.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)

#Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

##############################
#%%
#Exercise 2: 
#SVM With Randomly Generated Dataset

from sklearn.datasets import make_classification
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Create a dataset
X,y=make_classification(n_samples=50,n_features=2, n_redundant=0,
                        n_informative=2,random_state=2, n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

#Implement a cross-validation procedure to select the best hyperparameters for
#a linear classifier
svc_lin = SVC(kernel="linear")
parameters={'C':[i for i in range(1,11)]}
clf_lin = GridSearchCV(svc_lin,parameters)
clf_lin.fit(X,y)
best_lin = clf_lin.best_params_
print(best_lin.get('C'))

#Plot the decision surface 
h = .02 # grid step
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

#the grid is created, the intersections are in xx and yy
mysvc= SVC(kernel="linear", C=best_lin.get('C'))
mysvc.fit(X,y)
mysupp=mysvc.support_vectors_
Z2d = mysvc.predict(np.c_[xx.ravel(),yy.ravel()]) # we predict all the grid
Z2d=Z2d.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)

# We plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.scatter(mysupp[:, 0], mysupp[:, 1], color='black', marker='2')
plt.show()

###########
#and rbf-kernel
svc_rbf = SVC(kernel="rbf")
parameters={'C':[i for i in range(1,11)],'gamma':[j for j in range(0,11)]}
clf_rbf = GridSearchCV(svc_rbf,parameters)
clf_rbf.fit(X,y)
best_rbf = clf_rbf.best_params_
print(best_rbf)

#Plot the decision surface 
h = .02 # grid step
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

#the grid is created, the intersections are in xx and yy
mysvc= SVC(kernel="rbf",C=best_rbf.get('C'),gamma=best_rbf.get('gamma'))
mysvc.fit(X,y)
mysupp=mysvc.support_vectors_
Z2d = mysvc.predict(np.c_[xx.ravel(),yy.ravel()]) # we predict all the grid
Z2d=Z2d.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)

# We plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.scatter(mysupp[:, 0], mysupp[:, 1], color='black', marker='2')
plt.show()

##############################
#%%
#Exercise 3: 
#SVM With Moon Dataset

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
import random
import matplotlib.pyplot as plt

#Create dataset
X,y=make_moons(noise=0.1,random_state=1,n_samples=40)

#Train/Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())

#we create a mesh to plot in
plt.scatter(X[:,0],X[:,1],c=y,s=100)

#grid step
h = .02
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

#the grid is created, the intersections are in xx and yy
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))

#Create SVM model with linear kernal
mysvc= SVC(kernel='linear', C = 2.0)

#Train model with training split
mysvc.fit(X_train,Y_train)

#Predict on test
Z2d = mysvc.predict(X_test)
#print(Z2d.shape)
#print(xx.shape)
#Z2d=Z2d.reshape(xx.shape)
plt.figure()
#plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)

#We plot also the training points
print(np.sum(Z2d==Y_test))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.show()


##############################
#%%
#Exercise 4: 
#SVM With Iris Dataset

#Load Iris Dataset
from sklearn.datasets import load_iris
irisData = load_iris()
X = irisData.data[:,0:4]
y = irisData.target

#Train/test split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())

#SVM with linear kernal
mysvc= SVC(kernel='linear', C = 2.0)

#Train model
mysvc.fit(X_train,Y_train)

#Predict
Z2d = mysvc.predict(X_test)
#print(Z2d.shape)
#print(xx.shape)
#Z2d=Z2d.reshape(xx.shape)
#plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)

#Total test observations
print(len(Y_test))
#Observations we correctly predicted
print(np.sum(Z2d==Y_test))

##############################
#%%
#Exercise 5: 
#SVM With Ozone Data

import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
import random

#Read data
input_file = "ozone.dat"
oz = pd.read_csv(input_file,sep=" ")

#Split classes on Ozone threshold of 150 and use as label
y = oz.loc[:,"O3obs"]>150

'''
Station attribute contains categorical values that have to be converted.
Associating an integer to each station biases the dataset because some stations 
will gain more "importance" than others just because of the inte-ger choice. 
Better solution is replace each value by a binary vector containing only one 1. 
Example: Aix-en-Provence could be encoded as 1; 0; 0; 0; 0, 
Rambouillet as 0; 1; 0; 0; 0, Munchhausen as 0; 0; 1; 0; 0. Etc.
'''
#Make station feature binary
for elem in oz.loc[:,"STATION"].unique():
    oz.loc[:,elem] = oz.loc[:,'STATION']==elem
    oz[elem] = oz[elem].astype(int)

#Feature dataset
X = oz.drop(['O3obs','STATION'],axis=1) 

#Train/test split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())

#Create hyperparameters to iterate through and test 
#Kernal parameter:
kernel = np.array(['linear','rbf','sigmoid','poly','poly','poly','poly'])
#Degrees for polynomial kernels
degrees = np.array([np.NAN,np.NAN,np.NAN,2,3,4,5])
#Combing into DF
hyper = pd.DataFrame({'kernel':kernel,'degrees':degrees})
#%%

#Test hyperparameters to find SVM model with optimal loss
for index,row in hyper.iterrows():
    #Create SVM with specified hyperparameters
    if row['kernel'] == 'poly':
        oz_svc = SVC(kernel=row['kernel'], degree=row['degrees'], C = 2.0)
    else:
        oz_svc = SVC(kernel=row['kernel'], C = 2.0)
        
    #Train SVM
    oz_svc.fit(X_train,Y_train)
    
    #Evaluate with avg cross validation score
    hyper.loc[index,'cross_val'] = np.mean(cross_val_score(oz_svc,X_test, 
             Y_test,cv=5))

#%%
#Create SVM
np.mean(check)

hyper.loc[0,'cross_val'] = 1

#%%
for index,row in hyper.iterrows():
    print(index)
    print(row)
#%%
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


#%%
'''
Kernel=rbf
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import random
input_file="C:/Users/Usama Javaid/Documents/IPython Notebooks/ozone.dat"
df=pd.read_csv(input_file,sep=" ")
from sklearn.preprocessing import StandardScaler

y=df.loc[:,"JOUR"]>150
i=1
for elem in df.loc[:,"STATION"].unique():
    df.loc[df.loc[:,'STATION']==elem,"STATION"]=i
    i=i+1
df1=df[df.columns[[1,2,3,4,5,6,8,9]]]
df1 = StandardScaler().fit_transform(df[df.columns[[1,2,3,4,5,6,8,9]]])
df2=np.append(df[df.columns[[0]]],df1,axis=1)
df3=np.append(df[df.columns[[7]]],df2,axis=1)
df3=pd.DataFrame(df3)

y=df3.loc[:,1]
X=df3[df3.columns[[0,2,3,4,5,6,7,8,9]]]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())
mysvc= SVC(kernel='rbf', C = 2.0)
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


#%%
'''
Kernel=sigmoid
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import random
input_file="C:/Users/Usama Javaid/Documents/IPython Notebooks/ozone.dat"
df=pd.read_csv(input_file,sep=" ")
from sklearn.preprocessing import StandardScaler

y=df.loc[:,"JOUR"]>150
i=1
for elem in df.loc[:,"STATION"].unique():
    df.loc[df.loc[:,'STATION']==elem,"STATION"]=i
    i=i+1
df1=df[df.columns[[1,2,3,4,5,6,8,9]]]
df1 = StandardScaler().fit_transform(df[df.columns[[1,2,3,4,5,6,8,9]]])
df2=np.append(df[df.columns[[0]]],df1,axis=1)
df3=np.append(df[df.columns[[7]]],df2,axis=1)
df3=pd.DataFrame(df3)

y=df3.loc[:,1]
X=df3[df3.columns[[0,2,3,4,5,6,7,8,9]]]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())
mysvc= SVC(kernel='sigmoid', C = 2.0)
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


#%%
'''
Kernel=polynomial Degree=3
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import random
input_file="C:/Users/Usama Javaid/Documents/IPython Notebooks/ozone.dat"
df=pd.read_csv(input_file,sep=" ")
from sklearn.preprocessing import StandardScaler

y=df.loc[:,"JOUR"]>150
i=1
for elem in df.loc[:,"STATION"].unique():
    df.loc[df.loc[:,'STATION']==elem,"STATION"]=i
    i=i+1
df1=df[df.columns[[1,2,3,4,5,6,8,9]]]
df1 = StandardScaler().fit_transform(df[df.columns[[1,2,3,4,5,6,8,9]]])
df2=np.append(df[df.columns[[0]]],df1,axis=1)
df3=np.append(df[df.columns[[7]]],df2,axis=1)
df3=pd.DataFrame(df3)

y=df3.loc[:,1]
X=df3[df3.columns[[0,2,3,4,5,6,7,8,9]]]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())
mysvc= SVC(kernel='poly',degree=3, C = 2.0)
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


#%%
'''
Kernel=polynomial Degree=5
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import random
input_file="C:/Users/Usama Javaid/Documents/IPython Notebooks/ozone.dat"
df=pd.read_csv(input_file,sep=" ")
from sklearn.preprocessing import StandardScaler

y=df.loc[:,"JOUR"]>150
i=1
for elem in df.loc[:,"STATION"].unique():
    df.loc[df.loc[:,'STATION']==elem,"STATION"]=i
    i=i+1
df1=df[df.columns[[1,2,3,4,5,6,8,9]]]
df1 = StandardScaler().fit_transform(df[df.columns[[1,2,3,4,5,6,8,9]]])
df2=np.append(df[df.columns[[0]]],df1,axis=1)
df3=np.append(df[df.columns[[7]]],df2,axis=1)
df3=pd.DataFrame(df3)

y=df3.loc[:,1]
X=df3[df3.columns[[0,2,3,4,5,6,7,8,9]]]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())
mysvc= SVC(kernel='poly',degree=5, C = 2.0)
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


#%%
'''
Kernel=polynomial Degree=2
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import random
input_file="C:/Users/Usama Javaid/Documents/IPython Notebooks/ozone.dat"
df=pd.read_csv(input_file,sep=" ")
from sklearn.preprocessing import StandardScaler

y=df.loc[:,"JOUR"]>150
i=1
for elem in df.loc[:,"STATION"].unique():
    df.loc[df.loc[:,'STATION']==elem,"STATION"]=i
    i=i+1
df1=df[df.columns[[1,2,3,4,5,6,8,9]]]
df1 = StandardScaler().fit_transform(df[df.columns[[1,2,3,4,5,6,8,9]]])
df2=np.append(df[df.columns[[0]]],df1,axis=1)
df3=np.append(df[df.columns[[7]]],df2,axis=1)
df3=pd.DataFrame(df3)

y=df3.loc[:,1]
X=df3[df3.columns[[0,2,3,4,5,6,7,8,9]]]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())
mysvc= SVC(kernel='poly',degree=2, C = 2.0)
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


#%%
'''
Kernel=polynomial Degree=4
'''
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import random
input_file="C:/Users/Usama Javaid/Documents/IPython Notebooks/ozone.dat"
df=pd.read_csv(input_file,sep=" ")
from sklearn.preprocessing import StandardScaler

y=df.loc[:,"JOUR"]>150
i=1
for elem in df.loc[:,"STATION"].unique():
    df.loc[df.loc[:,'STATION']==elem,"STATION"]=i
    i=i+1
df1=df[df.columns[[1,2,3,4,5,6,8,9]]]
df1 = StandardScaler().fit_transform(df[df.columns[[1,2,3,4,5,6,8,9]]])
df2=np.append(df[df.columns[[0]]],df1,axis=1)
df3=np.append(df[df.columns[[7]]],df2,axis=1)
df3=pd.DataFrame(df3)

y=df3.loc[:,1]
X=df3[df3.columns[[0,2,3,4,5,6,7,8,9]]]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=random.seed())
mysvc= SVC(kernel='poly',degree=4, C = 2.0)
X_train=np.array(X_train,dtype=np.float)
X_test=np.array(X_test,dtype=np.float)
Y_train=np.array(Y_train,dtype=np.float)
Y_test=np.array(Y_test,dtype=np.float)
mysvc.fit((X_train),(Y_train))
Z2d = mysvc.predict(X_test)
print(np.sum(Z2d==Y_test))
print(mysvc.score(X_test,Y_test))
scoresTrain=cross_val_score(mysvc,X_train,Y_train,cv=5)
scoresAll=cross_val_score(mysvc,np.array(df3,dtype=np.float),np.array(y,dtype=np.float),cv=5)


