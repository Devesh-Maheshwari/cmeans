# -*- coding: utf-8 -*-

"""
Created on Wed Jun 19 12:04:27 2019
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
l=[[randint(0,20),randint(0,1)] for i in range(20)]
data=np.asarray(l)
'''
df=pd.read_csv('C:\\Users\\banana.csv')
data=np.asarray(df)

y=data[:,-1]
x=data[:,:-1]

for i in range(0,len(y)):       #Converting the labels in 1 and -1
    if y[i]==2:
        y[i]=0
        
clf=svm.SVC(kernel='linear',C=1.0,gamma='scale')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
clf.fit(x,y)

#deciding the classes

class_plus=[x_train[i] for i in range(0,len(x_train)) if (y_train[i]==1)]

class_min=[x_train[i] for i in range(0,len(x_train)) if (y_train[i]==0)]

x_p_mean=(np.mean(class_plus))

x_m_mean=(np.mean(class_min))

list_maxplus=[x_p_mean-i for i in class_plus]

list_maxmin=[x_m_mean-i for i in class_min]

r_plus=np.max(np.abs(list_maxplus))
r_minus=np.max(np.abs(list_maxmin))

#compute s
Si=[]

for i in class_plus:
    Si.append(1-(np.abs(x_p_mean-i)/(r_plus+0.01)))
for i in class_min:
    Si.append(1-(np.abs(x_m_mean-i)/(r_minus+0.01)))    
#print(Si)

m,n = x_train.shape
   
Si=np.asarray(Si,dtype=object)

Si = Si.astype(np.double)
cvx_Si = matrix(Si)

y_train = y_train.reshape(-1,1) * 1.
X_dash=x_train*y_train

C=1 #regularization parameter

H=np.dot(X_dash,X_dash.T)

P=matrix(H)
q =matrix(-np.ones((m, 1)))
g1 = np.asarray(np.diag(np.ones(m) * -1))           # −αi≤0
g2 = np.asarray(np.diag(np.ones(m)))
print(g1.shape,g2.shape)                # αi≤ Si*C
G = matrix(np.append(g1, g2, axis=0))
print(G.size)
h1=np.asarray(np.zeros(m))                          #matrix of 0
h2=np.matrix((cvx_Si*C))                            #Si*C
h = matrix(np.column_stack((h1,h2)).reshape(-1,1))
A = matrix(y.reshape(1,-1))
b = matrix(np.zeros(1))
print()
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10

sol=solvers.qp(P,q,G,h,A,b)
alpha=np.array(sol['x'])

w = np.dot((y_train * alpha).T ,x_train).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alpha > 1e-4).flatten()

#Computing b
b = y_train[S] - np.dot(x_train[S], w)

#Display results
print('Alphas = ',alpha[alpha > 1e-4])
print('w = ', w)
print('b = ', b[0])

y_pred = clf.predict(x_test)

yy=np.dot(w,x_test.T)+b
we=(yy.flatten())
#print(len(yy))
#for i in range(0,len(we))
we=yy[0]
for i in range(0,len(we)):
    
    if we[i]<0:
        we[i]=0
    else:
        we[i]=1
        
w=(clf.coef_)

b=clf.intercept_
print('w = ',w)
print('b = ',b)

print('computed Accuracy  : ',metrics.accuracy_score(y_test,we))
print('Accuracy using rbf kernel model : ',metrics.accuracy_score(y_test,y_pred))