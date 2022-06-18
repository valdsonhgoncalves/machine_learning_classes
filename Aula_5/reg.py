import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

    
features = np.loadtxt('features.txt')
labels = np.loadtxt('labels.txt')

x_train, x_test, y_train, y_test = train_test_split(features,labels, random_state=0)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

linreg = LinearRegression()
linreg.fit(x_train,y_train)

logreg = LogisticRegression(max_iter=1000,random_state=0)
logreg.fit(x_train,y_train)

print('[0]Linear Regression Training Accuracy: {}', linreg.score(x_train,y_train))
print('[1]Logistic Regression Training Accuracy: {}', logreg.score(x_train,y_train))

predict1 = linreg.predict(x_test)
# predict1[predict1>=0.5] = 1.0
# predict1[predict1<0.5] = 0.0
predict2 = logreg.predict(x_test)

print('[0]Linear Regression Test Accuracy: {}'.format(linreg.score(x_test,y_test)))
print('[1]Logistic Regression Test Accuracy: {}'.format(logreg.score(x_test,y_test)))

print('Model 1 prediction')
print(predict1)
print('-'*70)
print('Model 2 prediction')
print(predict2)
print('-'*70)
print('Real classification')
print(y_test)
        




