from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

#importar o conjunto de dados MNIST
dataset = fetch_openml("mnist_784")
(data,labels) = (dataset.data,dataset.target)

#Exibir algumas informações do conjunto
print("[INFO] Número de imagens: ", data.shape[0])
print("[INFO] Pixels por Imagem: ", data.shape[1])

#Mostrar uma imagem
np.random.seed(17)
randomindex = np.random.randint(0, data.shape[0])
plt.imshow(data[randomindex].reshape((28,28)),cmap="Greys")
plt.show()

#Normalizar as features no intervalo [0,1]
data = data.astype(float)/255.0

np.save("data.npy",data)
np.save("labels.npy",labels)
#dividir o conjunto de dados
(trainX,testX,trainY,testY) = train_test_split(data,dataset.target) #Default: 75% para treinamento e 25% para teste

#Converter labels inteiros para vetores
lb  = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#Definir a arquitetura de rede
model = Sequential()
model.add(Dense(128,input_shape=(784,),activation="sigmoid"))
model.add(Dense(64,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))
model.compile(optimizer=SGD(0.01),loss="categorical_crossentropy",metrics=["accuracy"])
H = model.fit(trainX,trainY,batch_size=128,epochs=100,verbose=2,validation_data=(testX,testY))

#Avaliar a rede neural
print("[INFO] Avaliando a Rede Neural")
predictions = model.predict(testX)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))

#Plot
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
