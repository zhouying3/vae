# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:09:52 2018

@author: zhouying
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives,optimizers
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(1111)  # for reproducibility

batch_size = 100 
n = 784
m = 20
hidden_dim = 500
epochs = 10
epsilon_std = 1.0
use_loss = 'mse' # 'mse' or 'xent'

decay = 0 # weight decay, a.k. l2 regularization
use_bias = True
initial = 'glorot_normal'
## Encoder
x = Input(batch_shape=(batch_size, n))
h_encoded = Dense(hidden_dim,kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='tanh')(x)
z_mean = Dense(m, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)(h_encoded)
z_log_var = Dense(m, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)(h_encoded)


## Sampler
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal_variable(shape=(batch_size, m), mean=0.,
                                       scale=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(m,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(hidden_dim, kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias, activation='tanh')
decoder_mean = Dense(n,kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)
decoder_std = Dense(n,kernel_regularizer=l2(decay), bias_regularizer=l2(decay), use_bias=use_bias)

def resampling(args):
    x_mean,x_logvar = args
    epsilon = K.random_normal_variable(shape=(batch_size,n),mean=0.,
                              scale = epsilon_std)
    return x_mean + K.exp(x_logvar / 2) * epsilon
## Decoder
h_decoded = decoder_h(z)
x_hat = Lambda(resampling, output_shape=(n,))([decoder_mean(h_decoded), decoder_std(h_decoded)])


## loss
def vae_loss(x, x_hat):
    kl_loss =  -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    xent_loss = n * objectives.binary_crossentropy(x, x_hat)
    mse_loss = n * objectives.mse(x, x_hat) 
    if use_loss == 'xent':
        return xent_loss + kl_loss
    else:
        return mse_loss + kl_loss
    

vae = Model(x, x_hat)
opt = optimizers.Adam(lr=1e-3)
vae.compile(optimizer=opt, loss=vae_loss)

# train the VAE on MNIST digits
mnist = input_data.read_data_sets('MNIST_data')
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
#(x_train, y_train), (x_test, y_test) = mnist

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#x_train -= np.mean(x_train,axis=0)
#x_test -= np.mean(x_train,axis=0)

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

##----------Visualization----------##
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
fig = plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()
#fig.savefig('z_{}.png'.format(use_loss))

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(m,))
_h_decoded = decoder_h(decoder_input)
_x_hat = decoder_mean(_h_decoded)

generator = Model(decoder_input, _x_hat)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
#fig.savefig('x_{}.png'.format(use_loss))

# data imputation
figure = np.zeros((digit_size * 3, digit_size * n))
x = x_test[:batch_size,:]
x_corupted = np.copy(x)
x_corupted[:, 300:400] = 0
x_encoded = vae.predict(x_corupted, batch_size=batch_size).reshape((-1, digit_size, digit_size))
x = x.reshape((-1, digit_size, digit_size))
x_corupted = x_corupted.reshape((-1, digit_size, digit_size))
for i in range(n):
    xi = x[i]
    xi_c = x_corupted[i]
    xi_e = x_encoded[i]
    figure[:digit_size, i * digit_size:(i+1)*digit_size] = xi
    figure[digit_size:2 * digit_size, i * digit_size:(i+1)*digit_size] = xi_c
    figure[2 * digit_size:, i * digit_size:(i+1)*digit_size] = xi_e

fig = plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
#fig.savefig('i_{}.png'.format(use_loss))