from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
 
#carregando dados de treino e teste
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # passando tudo para um vetor coluna
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    #transformando de alvo numérico para alvo categórico
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
 
#normalizando os pixeis
def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    #como o valor de pixel vai de 0-255, divide por 255 para que vá de 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm
 
#criar a estrutura da rede
def define_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add( Dropout(0.5) )

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    
	# compila o modelo
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
#Avaliar o modelo usando k-fold e cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepara pra cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        # define o modelo
        model = define_model()
        #seleciona os grupos de treino e teste
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        #ajusta o modelo
        history = model.fit(trainX, trainY, epochs=30, batch_size=32, validation_data=(testX, testY), verbose=0)
        #avalia o modelo
        _, acc = model.evaluate(testX, testY, verbose=1)
        print('> %.3f' % (acc * 100.0))
        #salvando os scores
        scores.append(acc)
        histories.append(history)
    return scores, histories
 
# plota curvas de aprendizado
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='b', label='train')
        plt.plot(histories[i].history['val_loss'], color='y', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='b', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='y', label='test')
    plt.show()
 
#resume a performance do modelo
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()
 


trainX, trainY, testX, testY = load_dataset()
trainX, testX = prep_pixels(trainX, testX)
scores, histories = evaluate_model(trainX, trainY)
summarize_diagnostics(histories)
summarize_performance(scores)