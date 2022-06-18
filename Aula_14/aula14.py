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

x_list = scipy.io.loadmat('DataForClassification_TimeDomain.mat')
x_mat = np.array(x_list['AccTimeDomain'])


labels = np.ones(104)

for i in range (2,10):
    labels = np.concatenate((labels, i*np.ones(104)))

x_train,x_resto,y_train,y_resto = train_test_split(np.transpose(x_mat),labels,test_size=0.30, random_state=None)

x_test, x_val, y_test, y_val = train_test_split(x_resto,y_resto,test_size=0.5,random_state=None)

# features
x_base = np.transpose(x_train)
x_test = np.transpose(x_test)
x_val = np.transpose(x_val)

# rms
feat_1 = np.std(x_base,axis=0) #válido para média zero
feat_1_test = np.std(x_test,axis=0)
feat_1_val = np.std(x_val,axis=0)

feat_2 = st.kurtosis(x_base,axis=0)
feat_2_test = st.kurtosis(x_test,axis=0)
feat_2_val = st.kurtosis(x_val,axis=0)

features = np.concatenate((feat_1.reshape(-1,1),feat_2.reshape(-1,1)),axis=1)
features_test = np.concatenate((feat_1_test.reshape(-1,1),feat_2_test.reshape(-1,1)),axis=1)
features_val = np.concatenate((feat_1_val.reshape(-1,1),feat_2_val.reshape(-1,1)),axis=1)

lb  = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
y_val = lb.transform(y_val)

model = Sequential()
model.add(Dense(2,input_shape=(2,),activation="sigmoid"))
model.add(Dense(4,activation="sigmoid"))
model.add(Dense(9,activation="softmax"))
model.compile(optimizer=SGD(0.01),loss="categorical_crossentropy",metrics=["accuracy"])
H = model.fit(features,y_train,batch_size=128,epochs=100,verbose=2,validation_data=(features_val,y_val))

print("[INFO] Avaliando a Rede Neural")
predictions = model.predict(features_val)
print(classification_report(y_val.argmax(axis=1),predictions.argmax(axis=1)))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,100),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,100),H.history["val_accuracy"],label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# box = np.reshape(feat_2,(104,9),order='F')
# plt.boxplot(box)
# plt.show()
#plt.scatter(feat_1,feat_2,c=y_train)



# plt.plot(x_mat[:,1])
 #plt.show()

i=1 