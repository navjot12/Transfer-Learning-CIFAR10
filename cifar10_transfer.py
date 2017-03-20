
# coding: utf-8

# In[29]:

import numpy as np
from matplotlib import pyplot as plt

import keras
from keras.datasets import cifar10
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils


# In[4]:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_examples = 50000


# In[7]:

X1_train = []
X1_test = []
X2_train = []
X2_test = []
Y1_train = []
Y1_test = []
Y2_train = []
Y2_test = []

for ix in range(n_examples):
    if y_train[ix] < 5:
        # put data in set 1
        X1_train.append(x_train[ix]/255.0)
        Y1_train.append(y_train[ix])
    else:
        # put data in set 2
        X2_train.append(x_train[ix]/255.0)
        Y2_train.append(y_train[ix])

for ix in range(y_test.shape[0]):
    if y_test[ix] < 5:
        # put data in set 1
        X1_test.append(x_test[ix]/255.0)
        Y1_test.append(y_test[ix])
    else:
        # put data in set 2
        X2_test.append(x_test[ix]/255.0)
        Y2_test.append(y_test[ix])


# In[13]:

X1_train = np.asarray(X1_train).reshape((-1, 32, 32, 3))
X1_test = np.asarray(X1_test).reshape((-1, 32, 32, 3))
X2_train = np.asarray(X2_train).reshape((-1, 32, 32, 3))
X2_test = np.asarray(X2_test).reshape((-1, 32, 32, 3))

Y1_train = np_utils.to_categorical(np.asarray(Y1_train), nb_classes=5)
Y1_test = np_utils.to_categorical(np.asarray(Y1_test), nb_classes=5)

Y2_train = np_utils.to_categorical(np.asarray(Y2_train), nb_classes=10)
Y2_test = np_utils.to_categorical(np.asarray(Y2_test), nb_classes=10)


# In[18]:

print X1_train.shape, X1_test.shape
print Y1_train.shape, Y1_test.shape


# In[26]:

split1 = int(0.8 * X1_train.shape[0])
split2 = int(0.8 * X2_train.shape[0])

x1_val = X1_train[split1:]
x1_train = X1_train[:split1]
y1_val = Y1_train[split1:]
y1_train = Y1_train[:split1]

x2_val = X2_train[split2:]
x2_train = X2_train[:split2]
y2_val = Y2_train[split2:]
y2_train = Y2_train[:split2]


# In[30]:

model = Sequential()

model.add(Convolution2D(32, 5, 5, input_shape=(32, 32, 3), activation='relu'))
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.42))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

import time, datetime

start = datetime.datetime.now()
hist1 = model.fit(x1_train, y1_train,
         nb_epoch=10,
         shuffle=True,
         batch_size=100,
         validation_data=(x1_val, y1_val), verbose=2)

time_taken = datetime.datetime.now() - start
print '\n'*2, '-'*20, '\n'
print 'Time taken for first training: ', time_taken
print '\n', '-'*20, '\n'*2


# In[31]:

for l in model.layers[:6]:
    l.trainable = False   

# In[ ]:

trans_model = Sequential(model.layers[:6])

trans_model.add(Dense(128))
trans_model.add(Activation('relu'))
trans_model.add(Dense(10))
trans_model.add(Activation('softmax'))

trans_model.summary()
trans_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

start = datetime.datetime.now()
hist2 = trans_model.fit(x2_train, y2_train, nb_epoch=10, shuffle=True, batch_size=100, validation_data=(x2_val, y2_val), verbose=2)
time_taken = datetime.datetime.now() - start
print '\n'*2, '-'*20, '\n'
print 'Time taken for final training: ', time_taken
print '\n', '-'*20, '\n'*2

