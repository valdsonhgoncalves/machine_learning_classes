from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
from scipy.optimize import minimize

def objective(x):
    print('x recebido =', x)
    clf = svm.SVC( C = x[0], kernel = 'poly' , degree = int(x[1]), gamma = x[2], max_iter= 1000000, verbose= False)
    clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train)
    print(np.log(scores.mean()/(scores.std() + 1e-6)))
    return np.log(scores.mean()/(scores.std() + 1e-6))

def constraint(x):
    return x[0]*x[1]*x[2] + x[0]*x[1] + x[0]*x[2] + x[1]*x[2] + x[0] + x[1] + x[2]


cancer = datasets.load_breast_cancer()


print("labels", cancer.target_names )
print('features',cancer.feature_names )

global x_train, x_test, y_train , y_test

x_train, x_test, y_train , y_test = train_test_split(cancer.data , cancer.target, test_size = 0.3, random_state = None )

#escolher modelo da SVM de classificação
clf = svm.SVC(C=1.0,kernel='rbf',degree=1,gamma='scale',coef0=0.0,shrinking=True,probability=False,tol=1e-3,class_weight=None,verbose=False,decision_function_shape='ovr',random_state=None)

#Ajustar o modelo
clf.fit(x_train,y_train)

#Estimar a confiança do Modelo
scores = cross_val_score(clf,x_train,y_train,cv=50)

#mostrar resultado
print("Acurácia Estimada = %6.2f (+/- %5.2f" %(scores.mean(),2*scores.std()))

#Testar
y_predict = clf.predict(x_test)
print("Acurácia testada = ",metrics.accuracy_score(y_test,y_predict))

#OTIMIZAÇÃO

# initial guesses
n = 3
x0 = np.zeros(n)
x0[0] = 5
x0[1] = 2
x0[2] = 1e-1

# show initial objective
print('Initial Objective: ' + str( objective(x0)  ))

# optimize
b = (1.0,5.0)

c_m = 0.1
c_p = 10
d_m = 1
d_p = 2
g_m = 1e-2
g_p = 1

bnds = ((c_m, c_p), (d_m, d_p) , (g_m, g_p) )
cons = {'type': 'eq', 'fun': constraint}
solution = minimize(objective,x0,method='SLSQP',bounds=bnds,constraints=cons)
x = solution.x

print ('y final',  objective(x))

# show final objective
print('Final Objective: ' + str( objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(int(x[1])))
print('x3 = ' + str(x[2]))
#print('x4 = ' + str(x[3]))

clf = svm.SVC( C = x[0], kernel = 'poly' , degree = int(x[1]), gamma = x[2], max_iter= 1000000, verbose= False)
clf.fit(x_train, y_train)

scores = cross_val_score(clf,x_train,y_train)
print("Acurácia Estimada = %6.2f (+/- %5.2f" %(scores.mean(),2*scores.std()))

y_predict = clf.predict(x_test)
print("Acurácia testada = ",metrics.accuracy_score(y_test,y_predict))