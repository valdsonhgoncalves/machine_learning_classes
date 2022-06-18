#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:05:09 2020

@author: alexandre
"""

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers

# plt.close("all")

### Importar dados MNIST:
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()    

# Reduzir número de dados no trienamento
x_train = x_train[:1000,:,:] 

# Normalização dos dados 
# Cada pixel tem 1 Byte = bite = 2⁸ = 256 
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255.
# Transformar imagens de 2D para 1D (Flatten)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[1], 1))
print(x_train.shape)
print(x_test.shape)

### ===================================
### Definir a arquitetura da rede neural autoencoder
### ===================================

# This is our input image
input_img = keras.Input(shape=(28,28,1))

# "encoded" é a representação compactada (espaço latente) do dado de entrada
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# "decoded" é a reconstrução da variável no espaço latente
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Modelo: mapear a imagem original (entrada) com a imagem reconstruída (saída)
autoencoder = keras.Model(input_img, decoded)

# # Modelo: mapear a imagem original (entrada) com a imagem compactada (meio)
# encoder = keras.Model(input_img, encoded)

# # This is our encoded (32-dimensional) input
# encoded_input = keras.Input(shape=(encoding_dim,))
# # Recuperar última camada do modelo da autoencoder
# decoder_layer = autoencoder.layers[-1]
# # Modelo: Mapear a imagem compactada (meio) com a saída da autoencoder
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

### ===================================
### Treinamento
### ===================================

# Compilar o modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Treinar o modelo
history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# Extrair imagens compactadas e imagens reconstruídas
# !!! Escolhidas do grupo de TESTE
# encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# Plotar curvas do bias e variância
bias = history.history['loss']
variance = history.history['val_loss']
plt.figure()
plt.plot(bias, label='Bias')
plt.plot(variance, label='Variância')
plt.legend()
plt.show()

print("Loss Training = ", bias[-1])
print("Loss Test = ", variance[-1])

ratio_bias_var = bias[-1]/variance[-1]

### ===================================
### Plotar imagens originais, compactadas e reconstruídas
### ===================================

# Número de dígitos a serem plotados
n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    # index = np.random.randint(0,x_test.shape[0],1)
    index = i
    
    # Imagem original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[index].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imagem reconstruída
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[index].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

