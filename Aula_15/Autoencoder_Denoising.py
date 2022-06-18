#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:59:35 2020

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
# x_train = x_train[:1000,:,:] 

# Normalização dos dados 
# Cada pixel tem 1 Byte = bite = 2⁸ = 256 
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255.
# Transformar imagens de 2D para 1D (Flatten)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]**2))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]**2))
print(x_train.shape)
print(x_test.shape)

# Criar imagens com ruído
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

### Definir dimensão da camada central da autoencoder
encoding_dim = 32  # Fator de compressão = 784/encoding_dim

### ===================================
### Definir a arquitetura da rede neural autoencoder
### ===================================

# Input image
input_img = keras.Input(shape=(784,))
# "encoded" é a representação compactada (espaço latente) do dado de entrada
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
                       # Adicionarregularização L1/L2
                       # activity_regularizer=keras.regularizers.l2(10e-5))(input_img)
# "decoded" é a reconstrução da variável no espaço latente
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# encoded = layers.Dense(128, activation='relu')(input_img)
# encoded = layers.Dense(64, activation='relu')(encoded)
# encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# # "decoded" é a reconstrução da variável no espaço latente
# decoded = layers.Dense(64, activation='relu')(encoded)
# decoded = layers.Dense(128, activation='relu')(decoded)
# decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Modelo: mapear a imagem original (entrada) com a imagem reconstruída (saída)
autoencoder = keras.Model(input_img, decoded)

# Modelo: mapear a imagem original (entrada) com a imagem compactada (meio)
encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Recuperar última camada do modelo da autoencoder
decoder_layer = autoencoder.layers[-1]
# Modelo: Mapear a imagem compactada (meio) com a saída da autoencoder
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

### ===================================
### Treinamento
### ===================================

# Compilar o modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Treinar o modelo
history = autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

# Extrair imagens compactadas e imagens reconstruídas
# !!! Escolhidas do grupo de TESTE
encoded_imgs = encoder.predict(x_test_noisy)
decoded_imgs = decoder.predict(encoded_imgs)

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

# Calcular número de pixels de cada lado da imagem compactada
# sqr_latent_dim = int(encoding_dim ** 0.5)
width_z = int(encoding_dim/4)
heigth_z = int(encoding_dim/(encoding_dim/4))

# Número de dígitos a serem plotados
n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    # index = np.random.randint(0,x_test.shape[0],1)
    index = i
    
    # Imagem original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[index].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imagem compactada 
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[index].reshape(width_z, heigth_z))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imagem reconstruída
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[index].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

