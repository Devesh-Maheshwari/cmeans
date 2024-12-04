# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:35:04 2019

@author: vb

link=https://www.openml.org/d/1460

"""

import skfuzzy as sk
import numpy as np
import pandas as pd
from cvxopt import solvers
from cvxopt import matrix
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from sklearn.model_selection import KFold

def kernel_mat(x_tr):
    A=np.matrix(x_tr)
    sigma=0.625
    
    m=A.shape[0]
    A=np.asarray(A)

    M=np.zeros((m,m))

    for i in range(0,len(A)):
        for j in range(0,len(A)):
            M[i,j]=gauss_mat(A[i],A[j],sigma)
    return M

def gauss_mat(xi,xj,v):
    return np.exp((-(np.linalg.norm(xi-xj)))/2*v**2)


def fcm(data):
    
    cnter, u, u0, Ed, Obj, itr, fpc = sk.cluster.cmeans(
        data, 2,2, error=0.01, maxiter=1000, init=None)

    return cnter

scores_kfcm=[]
scores_fsvm=[]
scores_svm=[]

sigma=0.625
c=10.0

df=pd.read_csv('C:\\Users\\nwoslab2\\Desktop\\satimage.csv')
orig_data=np.asarray(df)

y=orig_data[:6430,-1]
x=orig_data[:6430,:-1]


for i in range(0,len(y)):
    if y[i]==1 or y[i]==7:
        y[i]=1
    else:
        y[i]=0
  


#to avoid unknown value error in fit
y=y.astype('int') 

data= np.column_stack((x, y))

ctr=fcm(data).T
#Cross-Validation

clf=svm.SVC(kernel='rbf',C=1.0,gamma='scale')

kfold=KFold(10,True,24)

for train_index,test_index in kfold.split(ctr):
    
    x_train,x_test,y_train,y_test=(ctr[train_index],ctr[test_index],y[train_index],y[test_index])
    
    clf.fit(x_train,y_train)
    
    y_pred = clf.predict(x_test)





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
    
    M=kernel_mat(x_train)

    X_dash=M*y_train

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

    k=0.1

    b=(1/k)*(alpha*y_train)
     

    A=np.matrix(x_train)
    m=A.shape[0]
    A=np.asarray(A)
    A1=np.matrix(x_test)
    n=A1.shape[0]
    K_M=np.zeros((m,n))

    for i in range(0,len(A)):
        for j in range(0,len(A1)):
            K_M[i,j]=gauss_mat(A[i],A1[j],sigma)

    y_pred=(alpha*y_train)*K_M + b

    Y=y_pred[:,0]
    for i in range(0,len(Y)):
        if Y[i]>0:
            Y[i]=1.0
        else:
            Y[i]=0.0
    
    y_test=np.column_stack(y_train).ravel()
    y_pred = clf.predict(x_train)


    #end of processing data
    
    scores_svm.append(metrics.accuracy_score(y_train,y_pred))
    scores_kfcm.append(metrics.accuracy_score(y_test,Y))
   
acc_mean_svm=np.mean(scores_svm)
acc_sd_svm=np.std(scores_svm)

acc_mean_kfcm=np.mean(scores_kfcm)
acc_sd_kfcm=np.std(scores_kfcm)

print('---------SATIMAGE DATASET-----------',end='\n\n')

print('SVM Accuracy Mean : ',acc_mean_svm,end='\n')    
print('SVM Accuracy Standard Deviation : ',acc_sd_svm,end='\n\n')


print('KFCM-FSVM Accuracy Mean : ',acc_mean_kfcm,end='\n')    
print('KFCM-FSVM Accuracy Standard Deviation : ',acc_sd_kfcm,end='\n\n')

