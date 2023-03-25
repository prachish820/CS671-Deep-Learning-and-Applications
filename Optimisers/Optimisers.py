#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K


# In[2]:


# Loading MNIST data having images of handwritten digits [1,2,3,6,9]

root_dir = "./Group_13"
train_dir = os.path.join(root_dir,"train")
val_dir = os.path.join(root_dir,"val")
test_dir = os.path.join(root_dir,"test")


# In[3]:


def load_data(path):
    x,y = [],[]
    class_dirs = ["1","2","3","6","9"]
    
    for class_dir in class_dirs:
        class_path = os.path.join(path,class_dir)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path,img_file)
            
            #reading image
            with Image.open(img_path) as img:
                x.append(np.array(img))
                y.append(class_dir)
    x=np.array(x)
    y=np.array(list(map(int, y)))
    return x,y


# In[4]:


train_x,train_y=load_data(train_dir)
val_x,val_y=load_data(val_dir)
test_x,test_y=load_data(test_dir)


# In[5]:


# normalizing data
from sklearn.preprocessing import MinMaxScaler

train_x_flat=train_x.reshape(train_x.shape[0],-1)
val_x_flat=val_x.reshape(val_x.shape[0],-1)
test_x_flat=test_x.reshape(test_x.shape[0],-1)

scaler=MinMaxScaler()
scaler.fit(train_x_flat)
train_x=scaler.transform(train_x_flat)
val_x=scaler.transform(val_x_flat)
test_x=scaler.transform(test_x_flat)


# In[6]:


# set weight initializer for Keras layers
np.random.seed(1)
weights_ini=initializers.RandomNormal(mean=0,stddev=1,seed=123)
bias_ini=initializers.Zeros()


# In[23]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])


# In[24]:


model.summary()


# # SGD 

# In[25]:


# Setting optimiser
opt_SGD = keras.optimizers.SGD(learning_rate=0.001,name="SGD")

# compiling model
model.compile(optimizer=opt_SGD,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=1,epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=True)
loss_SGD=model_train.history['loss']
# saving model
model.save("./models/A1/SGD.h5")


# In[26]:


# plotting error vs epoch curve

plt.figure(figsize=(8,6))
plt.plot(loss_SGD,label='SGD')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
#locator = MaxNLocator(len(model_train.history['loss']), integer=True)
#plt.gca().xaxis.set_major_locator(locator)
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_SGD.png")
plt.show()
plt.close()

K.clear_session()


# # Batch Gradient Descent 

# In[27]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer
model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])

# Setting optimiser
opt_SGD = keras.optimizers.SGD(learning_rate=0.001,name="SGD")

# compiling model
model.compile(optimizer=opt_SGD,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=len(train_x),epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=False)
loss_batch = model_train.history['loss']
# saving model
model.save("./models/A1/batch.h5")


# In[28]:


# plotting error vs epoch curve

plt.figure(figsize=(8,6))
plt.plot(loss_batch,label='Vanilla')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_batch.png")
plt.show()
plt.close()

K.clear_session()


# # Momentum based Gradient Descent

# In[29]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])

# Setting optimiser
opt_momentum = keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,name="momentum_SGD")

# compiling model
model.compile(optimizer=opt_momentum,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=1,epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=True)
loss_mom = model_train.history['loss']
# saving model
model.save("./models/A1/mom.h5")


# In[31]:


# plotting error vs epoch curve
plt.figure(figsize=(8,6))
plt.plot(loss_mom,label='momentum')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
#locator = MaxNLocator(len(model_train.history['loss']), integer=True)
#plt.gca().xaxis.set_major_locator(locator)
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_mom.png")
plt.show()
plt.close()

K.clear_session()


# # NAG

# In[32]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])

# Setting optimiser
opt_NAG = keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True,name="NAG")

# compiling model
model.compile(optimizer=opt_NAG,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=1,epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=True)
loss_NAG = model_train.history['loss']

# saving model
model.save("./models/A1/NAG.h5")


# In[33]:


# plotting error vs epoch curve

plt.figure(figsize=(8,6))
plt.plot(loss_NAG,label='NAG')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_NAG.png")
plt.show()
plt.close()

K.clear_session()


# # Adagrad

# In[34]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])

# Setting optimiser
opt_Adagrad = keras.optimizers.Adagrad(learning_rate=0.001,epsilon=1e-08,name="Adagrad")

# compiling model
model.compile(optimizer=opt_Adagrad,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=1,epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=True)
loss_adagrad = model_train.history['loss']

# saving model
model.save("./models/A1/adagrad.h5")


# In[35]:


# plotting error vs epoch curve
plt.figure(figsize=(8,6))
plt.plot(loss_adagrad,label='Adagrad')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_adagrad.png")
plt.show()
plt.close()

K.clear_session()


# # RMSprop

# In[36]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])

# Setting optimiser
opt_RMSprop = keras.optimizers.RMSprop(learning_rate=0.001,rho=0.99,epsilon=1e-08,name="RMSprop")

# compiling model
model.compile(optimizer=opt_RMSprop,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=1,epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=True)
loss_RMSprop = model_train.history['loss']

# saving model
model.save("./models/A1/RMSprop.h5")


# In[37]:


# plotting error vs epoch curve
plt.figure(figsize=(8,6))
plt.plot(loss_RMSprop,label='RMSprop')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_RMSprop.png")
plt.show()
plt.close()

K.clear_session()


# # Adam

# In[38]:


# 3 layer FCNN with sigmoidal activation at hidden layers and softmax at output layer

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])

# Setting optimiser
opt_Adam = keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,name="Adam")

# compiling model
model.compile(optimizer=opt_Adam,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# setting convergence criterion
my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=1,verbose=1)

# fitting model
model_train=model.fit(train_x,train_y,batch_size=1,epochs=10000,callbacks=my_callbacks,verbose="auto",
         validation_data=(val_x,val_y),shuffle=True)
loss_adam = model_train.history['loss']

# saving model
model.save("./models/A1/adam.h5")


# In[39]:


# plotting error vs epoch curve
plt.figure(figsize=(8,6))
plt.plot(loss_adam,label='Adam')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
plt.legend()
plt.tight_layout()
plt.savefig("./models/A1/err_adam.png")
plt.show()
plt.close()

K.clear_session()


# # Superimposed plot of training error for all optimisers

# In[40]:


fig,ax=plt.subplots(figsize=(10,6))
ax.plot(loss_SGD,label='SGD')
ax.plot(loss_batch,label='Batch')
ax.plot(loss_mom,label='Momentum based')
ax.plot(loss_NAG, label='NAG')
ax.plot(loss_adagrad,label='Adagrad')
ax.plot(loss_RMSprop,label='RMSprop')
ax.plot(loss_adam,label='Adam')
plt.title("Average training error v/s epoch")
plt.xlabel('epoch')
plt.ylabel('Average train error')
plt.legend(loc='upper left',bbox_to_anchor=(1.02,1))
plt.tight_layout()
plt.savefig("./superimposed_plot.png")
plt.show()
plt.close()


# # Testing model on train, validation and test data for best selected architecture

# In[8]:


# loading best selected model

from keras import models

best_model=models.load_model("./models/A1/adam.h5")

# get weights of the model

weights=best_model.get_weights()

# create a Keras model with same architecture as best model and set its weights equal to the best model weights

model = keras.Sequential([
        layers.Flatten(input_shape=(784,),name='input_layer'),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_1"),
        layers.Dense(64,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_2"),
        layers.Dense(128,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="sigmoid",name="hidden_layer_3"),
        layers.Dense(10,kernel_initializer=weights_ini,bias_initializer=bias_ini,activation="softmax",name="output_layer")
        ])
# Setting optimiser ( select best optimiser by comparing model complexity and validation accuracy )
opt_Adam = keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,name="Adam")

model.set_weights(weights)

# compile the model

model.compile(optimizer=opt_Adam,loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[9]:


# testing model on train data

train_loss,train_acc=model.evaluate(train_x,train_y)

# predicting classes for train data

train_pred=model.predict(train_x,verbose=1)
pred_y=np.argmax(train_pred,axis=1)

# confusion matrix for train data

from tensorflow.math import confusion_matrix

conf_mat_train=confusion_matrix(train_y,pred_y)
print(conf_mat_train)


# In[10]:


# testing model on validation data for best architecture

val_loss,val_acc=model.evaluate(val_x,val_y)

# predicting classes for validation data for best architecture

val_pred=model.predict(val_x,verbose=1)
pred_y=np.argmax(val_pred,axis=1)

# confusion matrix for validation data

conf_mat_val=confusion_matrix(val_y,pred_y)
print(conf_mat_val)


# In[11]:


# testing model on test data

test_loss,test_acc=model.evaluate(test_x,test_y)

# predicting classes for test data

test_pred=model.predict(test_x,verbose=1)
pred_y=np.argmax(test_pred,axis=1)

# confusion matrix for test data

conf_mat_test=confusion_matrix(test_y,pred_y)
print(conf_mat_test)


# In[ ]:




