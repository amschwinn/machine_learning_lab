# -*- coding: utf-8 -*-
"""
Machine Learning Practical Session

Learning with Scikit-Learn
KNN, Decision Trees, MLP
Uses Iris dataset and original kayak dataset

By: Austin Schwinn

Feb 27, 2017
"""
#############################
#Exercise 1: 
#Warm up with IRISs

#Load the data
from sklearn.datasets import load_iris
irisData = load_iris()

#Print the data attributes
print(irisData.data)
print(irisData.target)
print(irisData.target_names)
print(irisData.feature_names)
print(irisData.DESCR)

#Execute the following commands and write down the
#answer in your report using the corresponding line number
#For each exercise

#Number of observations in the dataset. Returns 150.
len(irisData.data)

#Summary of the function. What the function does, the arguments, 
#and output type.
help(len)

#Name of the first plant type. Returns ‘setosa’
irisData.target_names[0]

#Name of the 3rd plant type. Returns ‘virginica’
irisData.target_names[2]

#Gives the first to last name from the name array, which is the 3rd plant type. 
#Returns ‘virginica’
irisData.target_names[-1]

#Returns an index error because an array index ends at the length – 1.
irisData.target_names[len(irisData.target_names)]

#Returns (150L,4L) which is the dimensions of the data matrix. 
#It is 150 observations by 4 features.
irisData.data.shape

#Returns the 4 features of the first observation.
irisData.data[0]

#Same as irisData.data[0][1] but more used in Python
#Give the second feature of the first observation
irisData.data[0,1]

#2nd feature of the 2nd and 3rd observation
irisData.data[1:3,1]

#all of the observations for the 2nd feature.
irisData.data[:,1]

#Matplotlib
from matplotlib import pyplot as plt
X=irisData.data
Y=irisData.target
x=0
y=1

#creates a scatterplot of all the observations with the x axis being the 
#first attribute and the y axis being second attribute. The points are colored 
#according to their target value (type of plant).
plt.scatter(X[:,x],X[:,y],c=Y)

#Renders the scatterplot created in previous line.
plt.show()

#summary of the function. What the function does, the arguments, 
#and the output type.
help(plt.scatter)

#Assigns the first feature name as the label for the x-axis (sepal length(cm))
plt.xlabel(irisData.feature_names[x])

#Assigns the second feature as the label for the y-axis (sepal width(cm))
plt.ylabel(irisData.feature_names[y])
#Assigns the second feature as the label for the y-axis (sepal width(cm))

#Fills in the scatterplot datapoints for the new plot
plt.scatter(X[:,x],X[:,y],c=Y)

#Renders the new plot
plt.show()

#Obtain a slightly more precise answer
#Gives a Boolean return if all the observations in the array equal the values. 
#For this instance, it is TRUE/FALSE if each observation of class (plant type) 
#is equal to 0.
Y==0

#Returns an array of arrays that are 4 feature observations that have the 
#class equal to 0. So in this instance, it is all the observations (displayed 
#in arrays of the 4 features) that have the specified class.
X[Y==0]

#Returns just the first feature of all the observations that have a 
#class type of 0.
X[Y==0][:,x]

#Creates a scatterplot of just feature 1 (x-axis) and 2 (y-axis) with the first 
#class type and labels the points red
plt.scatter(X[Y==0][:,x],X[Y==0][:,y],color="red",label=irisData.target_names[0])

#Creates a scatterplot of just feature 1 (x-axis) and 2 (y-axis) with the 
#second class type and labels the points green
plt.scatter(X[Y==1][:,x],X[Y==1][:,y],color="green",label=irisData.target_names[1])

#Creates a scatterplot of just feature 1 (x-axis) and 2 (y-axis) with the third 
#class type and labels the points blue
plt.scatter(X[Y==2][:,x],X[Y==2][:,y],color="blue",label=irisData.target_names[2])

#Provides the legend for what color corresponds to what class of plant
plt.legend()
#Provides the legend for what color corresponds to what class of plant

#Renders the scatterplot
plt.show()


#################################
#Exercise 3
#KNN Classifier on IRIS
from sklearn import neighbors
nb_neighb = 15
help(neighbors.KNeighborsClassifier)
clf = neighbors.KNeighborsClassifier(nb_neighb)
help(clf.fit)

#This fit the model and gives information on the KNN classifier. 
#For this instance it tells us that it’s algorithm parameter is ‘auto’, 
#leaf size of 30, calculated metrics using minkowski, uniformed weight, 
#15 neighbors and a p value of 2
clf.fit(X,Y)

#Clf.predict is used to predict the class labels for 
#provided data, as well as the parameters.
help(clf.predict)

#It tells us that it predicts class type 0
print(clf.predict([[5.4,3.2,1.6,0.4]]))

#probability that the observation is for each class type. 
#There is 100% chance it is the first class and 0% chance that it is 
#class type 2 or 3.
print(clf.predict_proba([[5.4,3.2,1.6,0.4]]))

#mean accuracy for prediction in the model. 
#Our output is a mean accuracy of .986666 which tells us that our model is 
#very accurate.
print(clf.score(X,Y))

#Assign class predictions based on our  data to a new variable array.
Z = clf.predict(X)

#returns our data observations where our predicted class is not the same 
#as our actual observed class.
print(X[Z!=Y])

#Split training and test data
from sklearn.model_selection import train_test_split
import random 

#We split the data into train and test sets for both X and Y. 
#We use 70% in the training set and 30% in the test set. 
#We set a random seed so we can check the stability of our algorithm as we 
#train and test several times.
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#how many observations are in our data training set (105)
len(X_train)

#How many observations are in our test set (45)
len(X_test)

#Number of observations in the training set that are of class 0 (36)
len(X_train[Y_train==0])

#Number of observations in X training set that are of class 1 (37)
len(X_train[Y_train==1])

#Number of observations in X training set that are of class 2 (32)
len(X_train[Y_train==2])

#fitting the model on our training sets instead of our full data
clf=clf.fit(X_train,Y_train)

#Predicts the classes for our test dataset based on our model, which we fitted 
#with our training sets.
Y_pred=clf.predict(X_test)

#Create a confusion matrix from to see how our test predictions differ from 
#our actual test classes. The confusion matrix is a 3x3 and gives the number of 
#observations that correspond with what is predicted and what is actually 
#present in our test data.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
#We correctly classified all by 3 observations. 
#Class 0 was correctly classified 100% of the time. 
#Every time an observation was predicted to be class 1, it was correctly. 
#3 of the observations that were predicted to be class 2 were actually class 1 
#but we correctly classified 15 observations of class 2.

#Try again using the paramater shuffle=False. What happened?
#The data was not shuffled before we split them into batches. 
#This is another parameter to further randomize our dataset

from sklearn.model_selection import KFold
#With shuffle = True
kf = KFold(n_splits=10,shuffle=True)
for learn,test in kf.split(X):
    print("app: ",learn," test: ",test)
#With shuffle = False
kf = KFold(n_splits=10,shuffle=False)
for learn,test in kf.split(X):
    print("app: ",learn," test: ",test)
#The algorithm’s output becomes unstable. 
#It no longer is able to find an ideal value for K and says that the correct 
#number of groups is 1 and all the scores are 0.

##################################
#Exercise 4
#Decision Trees on IRIS
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random 

#set our clf variable to a decision tree ready to be fit and modeled
clf=tree.DecisionTreeClassifier()

#Fits the tree parameters to be the most accurate predicting the plant type 
#based on the data matrix of sepal width and length, etc.
clf=clf.fit(iris.data,iris.target)

#predicting the class (plant type) of the 51st observation based on all of the 
#features as it is ran through our fitted decision tree. It is of class 1.
print(clf.predict([iris.data[50,:]]))

#This is the accuracy of our decision tree to predict the type of plant based 
#on all our data features. It gives us a value of 1.0 which means we correctly 
#predict the type correctly on all our observations.
print(clf.score(iris.data,iris.target))

#outputs a file that contains all of the nodes and the decision rules for that 
#node for our decision tree.
tree.export_graphviz(clf,out_file='tree.dot')

####
#Default parameters
clf=tree.DecisionTreeClassifier()
clf=clf.fit(iris.data,iris.target)

#Export default tree
tree.export_graphviz(clf,out_file='tree_default.dot')

for i in range(9,2,-1):
    clf=tree.DecisionTreeClassifier(max_leaf_nodes=i)
    clf=clf.fit(iris.data,iris.target)
    
    #Give score
    print(str(i) + "max leafs score:")
    print(clf.score(iris.data,iris.target))
    
    #Export modified tree
    file_name = 'tree_' + str(i) + '.dot'
    tree.export_graphviz(clf,out_file=file_name)
'''
Our default tree contains 9 leaves, so there is no change when we limit the 
max number of leaf nodes to 9. As we decrease from there, it does change. 
We continually go down the loop, each new tree matches the number of max_leafs. 
It is interesting to notice that the accuracy of the decision tree only goes 
down incrementally as we limit the max number of leaves. It one only goes from 
1.0 to .96, which is not as big of a drop off as I thought it would be.
'''
####
#Gini Train
clf=tree.DecisionTreeClassifier(criterion="gini")
clf=clf.fit(iris.data,iris.target)

#Export gini tree
tree.export_graphviz(clf,out_file='gini_iris.dot')

#Entropy Train
clf2=tree.DecisionTreeClassifier(criterion="entropy")
clf2=clf2.fit(iris.data,iris.target)

#Export entropy tree
tree.export_graphviz(clf2,out_file='entropy_iris.dot')
'''
The tree using the Gini criterion corresponds exactly with our default decision 
tree. I investigated to see why and it is because Gini is the default for the 
criterion function argument. The two trees have the exact same decision 
structure, same node leaves and branches. The only difference is that one uses 
entropy and one uses Gini. This would lead me to believe that our decision tree 
is stable.
'''
####
#make data set
X,Y=make_classification(n_samples=100000,n_features=20,n_informative=15,n_classes=3)

#split into test and training
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#make decision tree with i
for i in range(1,21):
    #create decision tree
    clf=tree.DecisionTreeClassifier(max_leaf_nodes=500*i)
    clf=clf.fit(X_train,Y_train)
    
    #print scores
    #training
    print("TRAIN: " + str(i)+ "*500" + " max leafs TRAIN score:")
    print("%6.4f" %clf.score(X_train,Y_train))
    #testing
    print("TEST: " + str(i)+ "*500" + " max leafs score:")
    print("%6.4f" %clf.score(X_test,Y_test))
'''
The first thing I notice is that the training set is always more accurate the 
testing set. This is known as overfitting. We train our model too perfectly to 
the training set that there is no room for variation from those specific 
observations. So when we move onto the testing set, it will not as accurately 
predict off of those observations. The second thing I notice is that once 
i>=14, the training accuracy was 1.0 and stayed constant. The test accuracy, 
on the other hand, fluctuated both up and down as it i>=14.
'''
####
#make data set
X,Y=make_classification(n_samples=100000,n_features=20,n_informative=15,n_classes=3)

#split into test and training
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#make decision tree with i
for i in range(1,41):
    #create decision tree
    clf=tree.DecisionTreeClassifier(max_depth=i)
    clf=clf.fit(X_train,Y_train)
    
    #print scores
    #training
    print("TRAIN: " + str(i)+ " max_depth score:")
    print("%6.4f" %clf.score(X_train,Y_train))
    #testing
    print("TEST: " + str(i)+ " max_depth score:")
    print("%6.4f" %clf.score(X_test,Y_test))
'''
At lower numbers of max depth, the training and test scores stay relatively 
close to each other. This tells me, that the model is not overfit at these 
lower max depth levels and that the accuracy from training is reliable. 
However, as we increase the max depth to the double digit range of >10, 
we see that the accuracy scores start to split with training becoming much 
more accurate than when we test. This is the range where our model starts to 
overfit. As the max depth >20, it becomes very clear that the model is overfit. 
Our training set converges on 100% accuracy while our actual test shows that 
peak just over 80% accuracy. 
'''

###########################################
#Exercise 5
#Neural Networks on Digits
from sklearn.datasets import load_digits
digits=load_digits()
digits.data[0]
digits.images[0]
digits.data[0].reshape(8,8)

#Look at pictures using Matplotlib
from matplotlib import pyplot as plt
plt.gray()
plt.matshow(digits.images[0])
plt.show()

#Count the number of examples of a particular class
Y=digits.target
print(len(Y[Y==0]))

#Use scikits's MLP
from sklearn.neural_network import MLPClassifier
X = digits.data

#First we prepare our MLP before we feed it our training examples. 
#Set a hidden layer of size 5 and one of size 2. Also, set the seed for the 
#random state as a value of 1. Then, we take the untrained model and fit it 
#with our digit data.
clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5, hidden_layer_sizes=(5,2),
                    random_state=1)
clf.fit(X,Y)

####
#Split into train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#Default model
clf=MLPClassifier(solver='lbfgs',
                    alpha=1e-5, hidden_layer_sizes=(5,2),
                    random_state=1)
clf=clf.fit(X_train,Y_train)
print("Default:")
print("%6.4f" %clf.score(X_test,Y_test))

#change learning rate model
clf=MLPClassifier(hidden_layer_sizes=(5,2),
                    random_state=1, learning_rate='adaptive')
clf=clf.fit(X_train,Y_train)
print("Adaptive learning rate:")
print("%6.4f" %clf.score(X_train,Y_train))

#change hidden layers model
clf=MLPClassifier(hidden_layer_sizes=(27,3),
                    random_state=1)
clf=clf.fit(X_train,Y_train)
print("27 hidden layesr:")
print("%6.4f" %clf.score(X_train,Y_train))

#change solver
clf=MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(5,2),
                    random_state=1)
clf=clf.fit(X_train,Y_train)
print("Solver=lbfgs:")
print("%6.4f" %clf.score(X_train,Y_train))

#change alpha
clf=MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,2),
                    random_state=1)
clf=clf.fit(X_train,Y_train)
print("Alpha=1e-5:")
print("%6.4f" %clf.score(X_train,Y_train))
'''
I can conclude that the default settings aren’t the most effective. 
Every time I changed a parameter, the accuracy improved. For each iteration, 
I changed one parameter. I changed the learning rate, hidden layers model, 
solver, and the alpha. With each separate change to the parameters, there was
 a .15 - .2 increase in the accuracy.
'''

###########################################
#Exercise 6
#Test KNN, DT, MLP on Kayak Data Set from EX2

from sklearn import neighbors
import pandas as p
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import LeaveOneOut
import numpy as np

#### 
## KNN

#Load data
kayak_data=p.read_csv('KayakRaces.csv')
X=kayak_data.ix[:,0:4]
Y=kayak_data.ix[:,4]

#train Knn with k=3 on my data
clf = neighbors.KNeighborsClassifier(3)
clf.fit(X, Y)

#Print Score
print("1) KK accuracy score with k=3")
print(clf.score(X,Y))
print("\n")

#Split into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#Train KNN with training subset
clf = neighbors.KNeighborsClassifier(3)
clf.fit(X_train, Y_train)

#Print a number of examples that are not well classified
Z = clf.predict(X_test)
print("2) Misclassified examples")
print(X_test[Z!=Y_test])
print("\n")

#Leave-one-out cross validation
X=np.asmatrix(X)
Y=np.asmatrix(Y).transpose()
loo = LeaveOneOut()
for train, test in loo.split(X):
    X_train=X[train]
    X_test=X[test]
    Y_train=Y[train]
    Y_test=Y[test]

#Train KNN with training subset
clf = neighbors.KNeighborsClassifier(3)
clf.fit(X_train, Y_train)

#Print Score
print("3) Leave one out cross validation accuracy:")
print(clf.score(X_test,Y_test))
print("\n")

####
## Decision Tree

import pandas as p
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import LeaveOneOut
from sklearn import tree


#Load data
kayak_data=p.read_csv('KayakRaces.csv')
X=kayak_data.ix[:,0:4]
Y=kayak_data.ix[:,4]

#train tree on my data
clf=tree.DecisionTreeClassifier()
clf.fit(X, Y)

#Print Score
print("1) Default decision tree")
print(clf.score(X,Y))
print("\n")
tree.export_graphviz(clf,out_file='tree_default.dot')

#Split into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#Train tree with training subset
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)

#Print a number of examples that are not well classified
Z = clf.predict(X_test)
print("2) Misclassified examples")
print(X_test[Z!=Y_test])
print("\n")
tree.export_graphviz(clf,out_file='tree_part2.dot')

#Leave-one-out cross validation
X=np.asmatrix(X)
Y=np.asmatrix(Y).transpose()
loo = LeaveOneOut()
for train, test in loo.split(X):
    X_train=X[train]
    X_test=X[test]
    Y_train=Y[train]
    Y_test=Y[test]

#Train tree with training subset
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)

#Print Score
print("3) Leave one out cross validation accuracy:")
print(clf.score(X_test,Y_test))
print("\n")
tree.export_graphviz(clf,out_file='tree_cross_validationt.dot')

####
## MLP

import pandas as p
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier

#Load data
kayak_data=p.read_csv('KayakRaces.csv')
X=kayak_data.ix[:,0:4]
Y=kayak_data.ix[:,4]

#train neural net with my data
clf=MLPClassifier(solver='lbfgs',
                    alpha=1e-5, hidden_layer_sizes=(5,2),
                    random_state=1)
clf.fit(X, Y)

#Print Score
print("1) neural net accuracy score")
print(clf.score(X,Y))
print("\n")

#Split into training and testing
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=random.seed())

#Train neural net with training subset
clf=MLPClassifier(solver='lbfgs',
                    alpha=1e-5, hidden_layer_sizes=(5,2),
                    random_state=1)
clf.fit(X_train, Y_train)

#Print a number of examples that are not well classified
Z = clf.predict(X_test)
print("2) Misclassified examples")
print(X_test[Z!=Y_test])
print("\n")

#Leave-one-out cross validation
X=np.asmatrix(X)
Y=np.asmatrix(Y).transpose()
loo = LeaveOneOut()
for train, test in loo.split(X):
    X_train=X[train]
    X_test=X[test]
    Y_train=Y[train]
    Y_test=Y[test]

#Train nueral net with training subset
clf=MLPClassifier(solver='lbfgs',
                    alpha=1e-5, hidden_layer_sizes=(5,2),
                    random_state=1)
clf.fit(X_train, Y_train)

#Print Score
print("3) Leave one out cross validation accuracy:")
print(clf.score(X_test,Y_test))
print("\n")

'''
Which algorithm can explain your concept the best?
The decision tree performed much better than the rest of the algorithms. 
It received an accuracy score of 1.0 in both the default question 1.0 tree 
and the leave one out cross validation tree. This tells me that the tree is 
stable and not overfit, since it performed well on both training and test data. 
This makes sense based on background knowledge of the data I used. 
My classifier was kayaking disciplines and 4 features to describe them. 
A decision tree is a good application for this structure. 
I believe that the data set does not contain enough to properly train the 
models. Both received an accuracy score of 0.0 with the LOO cross validation.
'''