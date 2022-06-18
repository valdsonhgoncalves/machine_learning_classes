import numpy as np
import matplotlib.pyplot as plt 
import scipy.io 
import scipy.stats as st
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from scipy import optimize
from scipy.optimize import minimize

class my_pnn:
    def __init__(self):
        pass

    def fit(self,x_train,y_train,x_val=[],y_val=[],s='sing', loss='l2'):
        self.loss_function = loss
        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)
        if x_val ==[]:
            self.x_val = np.copy(x_train)
            self.y_val = np.copy(y_train)
        else:
            self.x_val = np.copy(x_val)
            self.y_val = np.copy(y_val)
        self.MontaClasses(y_train)
        self.n_class = len(self.class_out)
        self.s = np.ones(self.n_class)
        if s == 'sing':
            self.fit_s_sing()
        elif s == 'mult':
            self.fit_s_mult()
        else:
            self.s *= 2*s*s
    
    def fit_s_sing(self):
        bounds = [(1e-6,10)]
        res = optimize.shgo(self.fob_sing,bounds)
        self.s = res.x*np.ones(self.n_class)

    def fob_sing(self,s):
        self.s = s*np.ones(self.n_class)
        y_pred = self.predict(self.x_val)
        if self.loss_function == 'l2':
            obj = np.sum((y_pred-self.y_val)*(y_pred-self.y_val))
        else:
            obj = - f1_score(y_pred,self.y_val,average='micro')
        return obj

    def fit_s_mult(self):
        bounds = [(1e-6,10)]
        for i in range(1,self.n_class):
            bounds += [(1e-6,10)]
        
        res = optimize.shgo(self.fob_mult,bounds)
        self.s = np.array(res.x)

    def fob_mult(self,s):
        self.s = np.array(s)
        y_pred = self.predict(self.x_val)
        if self.loss_function == 'l2':
            obj = np.sum((y_pred-self.y_val)*(y_pred-self.y_val))
        else:
            obj = - f1_score(y_pred,self.y_val,average='micro')
        return obj
    
    def predict(self,x_val):
        n = x_val.shape[0]
        pred = np.zeros((n,self.n_class))
        for i in range(self.n_class):
            x = np.array(self.class_in[i][0])
            for j in range(n):
                dif = -(x_val[j]-x)*(x_val[j]-x)
                pred[j,i] = np.sum(np.exp(dif/self.s[i]))
        
        return np.argmax(pred,axis=1)

    def MontaClasses(self,y_train):
        class_out = []
        class_in = []
        y_min = 0
        y_int = np.array(y_train,dtype=int)
        while 1 < 2:
            y_min = np.min(y_int)
            if y_min > 49999:
                break
            class_in += [[self.x_train[y_int==y_min]]]   
            class_out += [y_min]
            y_int[y_int==y_min] = 50000
        self.class_in = class_in
        self.class_out = class_out

x_list = scipy.io.loadmat("DataForClassification_TimeDomain.mat")
x_mat= np.array(x_list['AccTimeDomain'])
#plt.plot(x_mat[:,1])
#plt.show()
labels = np.ones(104)
for i in range(2,10) : 
    labels = np.concatenate( (labels , i*np.ones(104) ) )
labels -= 1
# features temporais com filtros
rms = np.std( x_mat , axis = 0 )
k = st.kurtosis( x_mat , axis = 0 )


par = np.zeros(( 2 , 936 ) )
par[0,:] = rms/np.max(rms)
par[1,:] = k /np.max(k)





# Separar os conjuntos
par = np.transpose( par )
features = par

#freq , dep_128 = scipy.signal.welch(x_mat, fs=20000, window='hann', nperseg=256, noverlap=128, nfft=256, 
#       detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')



#features = np.transpose( dep_128 )


# dividir o dataset entre train (75%) e test (25%)
(trainX, testx, trainY, testy) = train_test_split(features, labels, test_size=0.4,random_state=None)
(validX, testX, validY, testY) = train_test_split(testx, testy, test_size=0.5,random_state=None)

model = my_pnn()
model.fit(trainX, trainY,x_val=validX,y_val=validY,s='mult',loss='f1_score')

pred = model.predict(testX)


plt.plot(testY,'b')
plt.plot(pred,'r')
plt.show()
print('s = ', np.sqrt(model.s)/2)
print(classification_report(testY,pred))
