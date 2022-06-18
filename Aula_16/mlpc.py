import numpy as np
import matplotlib.pyplot as plt 
import scipy.io 
import scipy.stats as st
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


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

nyquist = 10000
band_pass = np.array([6500/nyquist,9000/nyquist])
filt = scipy.signal.butter(N=4, Wn=band_pass,btype='bandpass',analog=False,output='sos')
x_filter = scipy.signal.sosfilt(filt,x_mat,axis=0)
rms_filt_1 = np.std(x_filter,axis=0)



par = np.zeros(( 3 , 936 ) )
par[0,:] = rms/np.max(rms)
par[1,:] = k /np.max(k)
par[2,:] = rms_filt_1/np.max(rms_filt_1)





# Separar os conjuntos
par = np.transpose( par )
features = par

freq , dep_128 = scipy.signal.welch(x_mat, fs=20000, window='hann', nperseg=256, noverlap=128, nfft=256, 
       detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')

dep_128 = dep_128/np.tile(np.max(dep_128,axis=1).reshape(-1,1),(1,936))
features = np.transpose( dep_128 )


# dividir o dataset entre train (75%) e test (25%)
(trainX, testx, trainY, testy) = train_test_split(features, labels, test_size=0.4,random_state=None)
(validX, testX, validY, testY) = train_test_split(testx, testy, test_size=0.5,random_state=None)

model = MLPClassifier(hidden_layer_sizes=(18,),random_state=1,max_iter=300,solver='adam')
model.fit(trainX, trainY)

pred = model.predict(testX)


print(classification_report(testY,pred))
plt.plot(testY,'b')
plt.plot(pred,'r')
plt.show()
