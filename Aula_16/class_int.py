from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats as st
import scipy.signal

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

#freq , dep_128 = scipy.signal.welch(x_mat, fs=20000, window='hann', nperseg=256, noverlap=128, nfft=256, 
#       detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')



#features = np.transpose( dep_128 )


# dividir o dataset entre train (75%) e test (25%)
(trainX, testx, trainY, testy) = train_test_split(features, labels, test_size=0.4,random_state=None)
(validX, testX, validY, testY) = train_test_split(testx, testy, test_size=0.5,random_state=None)
# converter labels de inteiros para vetores
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
validY = lb.fit_transform(validY)
testY = lb.transform(testY)

# definir a arquitetura da Rede Neural usando Keras
# 784 (input) =&gt; 128 (hidden) =&gt; 64 (hidden) =&gt; 10 (output)
model = Sequential()
model.add(Dense(8, input_shape=(3,), activation="sigmoid"))
#model.add(Dense(64, activation="sigmoid"))
model.add(Dense(9, activation="softmax"))

# treinar o modelo usando SGD (Stochastic Gradient Descent)
print("[INFO] treinando a rede neural...")
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
             metrics=["accuracy"])
H = model.fit(trainX, trainY, batch_size=32, epochs=5000, verbose=2,
         validation_data=(validX, validY))
# avaliar a Rede Neural
print("[INFO] avaliando a rede neural...")
predictions = model.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

# plotar loss e accuracy para os datasets 'train' e 'test'
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,5000), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,5000), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,5000), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,5000), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()