import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pyDOE2

# Carregar o conjunto de dados
cancer = datasets.load_breast_cancer()

# print('labels',cancer.target_names)
# print('features',cancer.feature_names)

# Separar conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=None)

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# Escolher o modelo da SVM de classificação

clf = svm.SVC(C=1.0,kernel='rbf',degree=1,gamma='scale',coef0=0.0,shrinking=True,probability=False,tol=1e-3,class_weight=None,verbose=False,decision_function_shape='ovr',random_state=None)

#Ajustar o modelo
clf.fit(X_train,y_train)

#Estimar a confiança do Modelo
scores = cross_val_score(clf,X_train,y_train,cv=50)

#mostrar resultado
print("Acurácia Estimada = %6.2f (+/- %5.2f" %(scores.mean(),2*scores.std()))

#Testar
y_predict = clf.predict(X_test)
print("Acurácia testada = ",metrics.accuracy_score(y_test,y_predict))

#Otimização na raça

C_m = 0.1
C_p = 10
d_m = 1
d_p = 2
g_m = 1e-2
g_p = 1

fat_2 = pyDOE2.ff2n(3)
par = np.zeros([8,3])
par[:,0] = C_m
par[fat_2[:,0] > 0,0] = C_p
par[:,1] = d_m
par[fat_2[:,1] > 0,1] = d_p
par[:,2] = g_m
par[fat_2[:,2] > 0,2] = g_p
y = np.zeros(8)

for i in range(8):
    clf = svm.SVC(C = par[i,0],kernel='poly',degree=int(par[i,1]),gamma=par[i,2], max_iter=2000)
    clf.fit(X_train,y_train)
    scores = cross_val_score(clf,X_train,y_train)
    y[i] = np.log(scores.mean()/(scores.std() + 1e-6))

# y = a0 + a1*x1 + a2*x2 + a3*x3 + a4*x1*x2 + a5*x1*x2 + a6*x2*x3 + a7*x1*x2*x3

# Ajustar a regressão linear
um = np.ones(8).reshape(-1,1)
X = np.concatenate((um,fat_2),axis=1)

for i in range(2):
    for j in range(i+1,3):
        xixj = fat_2[:,i]*fat_2[:,j]
        X = np.concatenate((X,xixj.reshape(-1,1)),axis=1)

X = np.concatenate((X,(fat_2[:,0]*fat_2[:,1]*fat_2[:,2]).reshape(-1,1)),axis=1)
Xt = np.transpose(X)
a = np.linalg.inv(Xt@X)@Xt@y.reshape(-1,1)

a[1:8] *= 2
print(a)

clf = svm.SVC(C = 1.0,kernel='poly',degree=1,gamma=1e-2, max_iter=2000)
clf.fit(X_train,y_train)
scores = cross_val_score(clf,X_train,y_train)
print("Acurácia Estimada = %6.2f (+/- %5.2f" %(scores.mean(),2*scores.std()))

y_predict = clf.predict(X_test)
print("Acurácia testada = ",metrics.accuracy_score(y_test,y_predict))

# clf = LinearDiscriminantAnalysis()
# clf.fit(X_train,y_train)
# print("Acurácia Estimada = %6.2f (+/- %5.2f" %(scores.mean(),2*scores.std()))

# y_predict = clf.predict(X_test)
# print("Acurácia testada = ",metrics.accuracy_score(y_test,y_predict))

i= 1
