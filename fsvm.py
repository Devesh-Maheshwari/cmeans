# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:29:58 2019

@author: vb
"""

from cvxopt import matrix
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from cvxopt import solvers
import numpy as np
from random import randint
import pandas as pd

'''
dataset=['pima.csv','AUSTRALIA.csv','blood_transfusion.csv','bupa.csv',\
         'satimage.csv','banana.csv','sonar.csv','spambase.csv','abalone.csv'\
         ,'monk_train.csv']
'''

df=pd.read_csv('C:\\Users\\Desktop\\Python\\AUSTRALIA.csv')
data=np.asarray(df)
 
y=data[:,-1]        #labels
x=data[:,:-1]       #features


#Regularisation parameter
c=64
     
#SVM model
clf=svm.SVC(kernel='rbf',C=c,gamma='scale')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
clf.fit(x_train,y_train)

#dividing classes

class_plus=[x_train[i] for i in range(0,len(x_train)) if (y_train[i]==1)]

class_min=[x_train[i] for i in range(0,len(x_train)) if (y_train[i]==0)]

class_plus=np.matrix(class_plus)

class_plus=(class_plus.transpose())     #separating the features in different columns
 
class_min=np.matrix(class_min)

class_min=(class_min.transpose())       


mean_plus=[np.mean(i) for i in class_plus]  #mean of each column or feature of positive class
mean_min=[np.mean(i) for i in class_min]    #mean of each column or feature of negative class


#Difference of each feature from it's mean in +ve class
list_plus=[]
k=-1
for i in class_plus:
    k+=1
    ll=[]
    for j in i:
        ll.append(mean_plus[k]-j)
    list_plus.append(ll)


#Difference of each feature from it's mean in -ve class
list_min=[]
k=-1
for i in class_min:
    k+=1
    ll=[]
    for j in i:
        ll.append(mean_min[k]-j)
    list_min.append(ll)
    
abs_plus=[(np.abs(i)) for i in list_plus]   #Computing the absolute value of the differences
max_plus=[np.max(i) for i in abs_plus]      #Taking out the maximum value out of the absolute value of the features

abs_min=[(np.abs(i)) for i in list_min]     #Computing the absolute value of the differences
max_min=[np.max(i) for i in abs_min]        #Taking out the maximum value out of the absolute value of the features


r_plus=np.mean(max_plus)                    #mean of the maximum value of the features as radius of +ve class
r_min=np.mean(max_min)                      #mean of the maximum value of the features as radius of -ve class

class_plus=class_plus.T                     #separating the points again as it was earlier
max_xi_pl=[np.max(i) for i in class_plus]   #mean of features {eg: max(a1,b1,c1) where a , b ,c are 3 features} corresponding to each point
x_bar_pl=(np.max(max_plus))                 #Taking the maximum of the maximum values of the 3 features 

class_min=class_min.T
max_xi_min=[np.max(i) for i in class_min]
x_bar_min=(np.max(max_min))


#compute s fuzzy membership function
Si=[]

for i in max_xi_pl:
    Si.append(np.abs(1-(np.abs(x_bar_pl-i)/(r_plus+0.01))))     #taking delta as 0.01
for i in max_xi_min:
    Si.append(np.abs(1-(np.abs(x_bar_min-i)/(r_min+0.01))))    


Si=np.array(Si,dtype=object)

Si = Si.astype(np.double)   
cvx_Si = matrix(Si)


#computing alpha using cvxopt 
m,n = x_train.shape
   
y_train = y_train.reshape(-1,1) * 1.
X_dash=x_train*y_train

C=c #regularization parameter

#Hi,j=y(i)y(j)<x(i)x(j)>
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format

P=matrix(H)                             
q =matrix(-np.ones((m, 1)))

g1 = np.asarray(np.diag(np.ones(m) * -1))           # −αi≤0
g2 = np.asarray(np.diag(np.ones(m)))                # αi≤ Si*C
G = matrix(np.append(g1, g2, axis=0))

h1=np.asarray(np.zeros(m))                          #matrix of 0
h2=np.matrix((cvx_Si*C))                            #Si*C
h = matrix(np.column_stack((h1,h2)).reshape(-1,1))

A=matrix(np.array(y_train.T))                       #y.T*α=0
b = matrix(np.zeros(1))

#Setting solver parameters 
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10

#Running Solver
sol=solvers.qp(P,q,G,h,A,b)
alpha=np.array(sol['x'])

#Selecting the set of indices S corresponding to non zero alphas
S=(alpha>1e-4).flatten()
       
#Computing Weight Vector w
w = np.dot((y_train[S] * alpha[S]).T ,x_train[S])

#Computing Bias Term
b = y_train[S] - np.dot(x_train[S], w.T)
b=np.min(np.abs(b))

#Computing equation of y
yy=np.dot(w,x_train.T)+b

yy=yy[0]    #Converting matrix of the form [[]] to []    


#Predicted value from model
y_pred = clf.predict(x_train)

#Converting labels in the form of -1s and 1s
for i in range(0,len(yy)):
    if yy[i]<0:
        yy[i]=-1
    else:
        yy[i]=1

#Display result 
print('computed Accuracy  : ',metrics.accuracy_score(y_train,yy))
print('Accuracy using rbf kernel model : ',metrics.accuracy_score(y_train,y_pred))