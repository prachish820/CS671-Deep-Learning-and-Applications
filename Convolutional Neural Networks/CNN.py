#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# In[3]:


# Loading Caltech-101 image dataset with five classes

root_dir = ".\Group_13"
train_dir = os.path.join(root_dir,"train")
val_dir = os.path.join(root_dir,"val")
test_dir = os.path.join(root_dir,"test")


# In[4]:


class_map = {"brain":0,"buddha":1,"ketch":2,"Leopards":3,"scorpion":4}
def load_data(path):
    x,y = [],[]
    class_dirs = list(class_map.keys())

    for class_dir in class_dirs:
        class_path = os.path.join(path,class_dir)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path,img_file)
            
            #reading and resizing image
            img = cv2.imread(img_path)
            x.append(np.array(cv2.resize(img,(224,224))))
            y.append(class_map[class_dir])
    x=np.array(x)
    y=to_categorical(y,num_classes=5)
    
    return x,y


# In[5]:


train_x,train_y=load_data(train_dir)
val_x,val_y=load_data(val_dir)
test_x,test_y=load_data(test_dir)


# In[6]:


# normalizing data
from sklearn.preprocessing import MinMaxScaler

def preprocess(data):
    scaler=MinMaxScaler()
    channel=[]
    
    for i in range(3):
        channel.append(scaler.fit_transform(data[:,:,:,i].reshape(len(data),-1)).reshape(len(data),224,224))
    data_norm=np.stack(channel,axis=-1)

    return data_norm


# In[7]:


train_x_scaled=preprocess(train_x)
val_x_scaled=preprocess(val_x)
test_x_scaled=preprocess(test_x)


# In[8]:


print(train_x.shape,val_x.shape,test_x.shape)


# # *Building model*

# In[9]:


weights_ini1=initializers.HeNormal(seed=100)
weights_ini2=initializers.VarianceScaling(scale=1.0,mode="fan_in",distribution="untruncated_normal",seed=123)
bias_ini=initializers.Zeros()


# In[10]:


# input layer
inputs = Input(shape=(224,224,3),name='input_layer')

# convolution layer 1
h1 = Conv2D(filters=8,kernel_size=(11,11),strides=4,padding="valid",activation="relu",
            kernel_initializer=weights_ini1,bias_initializer=bias_ini,name='conv_layer1')(inputs)
h1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),name='maxpool_layer1')(h1)

# convolution layer 2
h2 = Conv2D(16,(5,5),1,activation="relu",kernel_initializer=weights_ini1,bias_initializer=bias_ini,
            name='conv_layer2')(h1)
h2 = MaxPooling2D((3,3),2,name='maxpool_layer2')(h2)

# Flatten layer
h3 = Flatten()(h2)

# Dense layers
h4 = Dense(128,activation='relu',kernel_initializer=weights_ini2,bias_initializer=bias_ini,
           name='dense_layer1')(h3)

# Output layer
outputs = Dense(5,activation='softmax',kernel_initializer=weights_ini2,bias_initializer=bias_ini,
                name='output_layer')(h4)

model = Model(inputs=inputs, outputs=outputs)


# In[11]:


model.summary()


# # *Compiling model*

# In[12]:


adam = keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,name="Adam")
model.compile(optimizer=keras.optimizers.Adam(),loss="categorical_crossentropy",metrics=["accuracy"])

my_callbacks = EarlyStopping(monitor='loss',min_delta=0.0001,patience=8,verbose=1)


# In[13]:


train_model = model.fit(train_x_scaled,train_y,batch_size=32,epochs=1000,callbacks=my_callbacks,validation_data=(val_x_scaled,val_y),shuffle=True)


# In[14]:


history = train_model.history

import pickle
with open('./models/CNN1/history.pkl','wb') as f:
    pickle.dump(history,f)

# plot loss function
plt.plot(history['loss'])
plt.ylabel('Average Error',fontsize=12)
plt.xlabel('Epoch',fontsize=12)
plt.savefig('./models/CNN1/loss.png')
plt.show()


# In[15]:


# saving model
model.save('./models/CNN1/model.h5')


# In[16]:


# evaluating model
train_loss,train_acc = model.evaluate(train_x_scaled,train_y)
val_loss,val_acc = model.evaluate(val_x_scaled,val_y)
test_loss,test_acc = model.evaluate(test_x_scaled,test_y)


# In[17]:


# predicting train, validation and test data labels
train_out = model.predict(train_x_scaled)
val_out = model.predict(val_x_scaled)
test_out = model.predict(test_x_scaled)


# In[18]:


# confusion matrix on train, validation and test data
from tensorflow.math import confusion_matrix

def conf_mat(true,predicted):
    true_label = np.argmax(true,axis=1)
    predicted_label = np.argmax(predicted,axis=1)
    mat = confusion_matrix(true_label,predicted_label)
    
    return mat


# In[19]:


print(conf_mat(train_y,train_out))
print(conf_mat(val_y,val_out))
print(conf_mat(test_y,test_out))


# In[41]:


img_idx = 200

plt.imshow(train_x_scaled[img_idx])
plt.axis('off')
plt.show()


# # Visualizing feature maps

# In[42]:


layer_name = 'conv_layer1'
hidden_layer = model.get_layer(layer_name).output

feature_model = Model(inputs=inputs,outputs=hidden_layer)

features = feature_model.predict(train_x[img_idx:img_idx+1])
print(features.shape)

fig, axs = plt.subplots(2,4,figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
    ax.imshow(features[0,:,:,i])
    ax.axis('off')
plt.savefig('./models/CNN1/'+layer_name+'.png')
plt.show()
plt.close()


# In[43]:


layer_name = 'conv_layer2'
hidden_layer = model.get_layer(layer_name).output

feature_model = Model(inputs=inputs,outputs=hidden_layer)

features = feature_model.predict(train_x[img_idx:img_idx+1])

fig, axs = plt.subplots(2,4,figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
    ax.imshow(features[0,:,:,2*i])
    ax.axis('off')
plt.savefig('./models/CNN1/'+layer_name+'.png')
plt.show()
plt.close()


# # Visualizing filters

# In[25]:


layer_name = 'conv_layer1'
filters = model.get_layer(layer_name).get_weights()[0]

# normalize filters
scaler = MinMaxScaler()
filters = scaler.fit_transform(filters.reshape(-1,1)).reshape(filters.shape)   

fig, axs = plt.subplots(2,4,figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
    f = filters[:,:,:,i]
    ax.imshow(f[:,:,0],cmap = 'gray')
    ax.axis('off')
plt.savefig('./models/CNN1/filters_'+layer_name+'.png')
plt.show()
plt.close()


# In[27]:


layer_name = 'conv_layer2'
filters = model.get_layer(layer_name).get_weights()[0]

# normalize filters
scaler = MinMaxScaler()
filters = scaler.fit_transform(filters.reshape(-1,1)).reshape(filters.shape)   

fig, axs = plt.subplots(2,4,figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
    f = filters[:,:,:,2*i]
    ax.imshow(f[:,:,0],cmap = 'gray')
    ax.axis('off')
plt.savefig('./models/CNN1/filters_'+layer_name+'.png')
plt.show()
plt.close()


# # Visualizing patches in images

# In[347]:


last_conv_layer = model.get_layer('conv_layer2')

img = np.expand_dims(train_x_scaled[0],axis=0)

def get_feature(layer_name,x):
    layer_out = [layer.output for layer in model.layers if layer.name == layer_name]
    activation_model = Model(inputs=model.input, outputs=layer_out)
        
    return activation_model.predict(x)


# In[348]:


last_conv_feature = get_feature('conv_layer2',img)
last_conv_feature.shape


# In[349]:


# maximally activated neuron across all channels
def get_max_act(feature):
    
    feature = np.squeeze(last_conv_feature)
    max_act_by_ch = np.amax(feature,axis=(0,1))
    max_ch = np.argmax(max_act_by_ch)
    max_act_neuron = np.argmax(feature[:,:,max_ch])
    
    return max_act_neuron,max_ch


# In[350]:


def trace_patch1(max_act):
    
    act = [(max_act[0]-2, max_act[1]-2), (max_act[0]-2, max_act[1]-1), (max_act[0]-2, max_act[1]), (max_act[0]-2, max_act[1]+1), (max_act[0]-2, max_act[1]+2),
           (max_act[0]-1, max_act[1]-2), (max_act[0]-1, max_act[1]-1), (max_act[0]-1, max_act[1]), (max_act[0]-1, max_act[1]+1), (max_act[0]-1, max_act[1]+2),
           (max_act[0], max_act[1]-2), (max_act[0], max_act[1]-1), (max_act[0], max_act[1]), (max_act[0], max_act[1]+1), (max_act[0], max_act[1]+2),
           (max_act[0]+1, max_act[1]-2), (max_act[0]+1, max_act[1]-1), (max_act[0]+1, max_act[1]), (max_act[0]+1, max_act[1]+1), (max_act[0]+1, max_act[1]+2),
           (max_act[0]+2, max_act[1]-2), (max_act[0]+2, max_act[1]-1), (max_act[0]+2, max_act[1]), (max_act[0]+2, max_act[1]+1), (max_act[0]+2, max_act[1]+2),]
    
    return act


# In[353]:


def trace_poolpatch1(max_act):
    
    filter_size = 3 
    max_act = (max_act[0]*2, max_act[1]*2)
    act = [(max_act[0], max_act[1]), (max_act[0], max_act[1]+1), (max_act[0], max_act[1]+2), 
           (max_act[0]+1, max_act[1]), (max_act[0]+1, max_act[1]+1), (max_act[0]+1, max_act[1]+2), 
           (max_act[0]+2, max_act[1]), (max_act[0]+2, max_act[1]+1), (max_act[0]+2, max_act[1]+2)]
    return act


# In[355]:


def trace_patch2(max_act):
    
    max_act = (max_act[0]*4, max_act[1]*4)
    act = [(max_act[0]-5, max_act[1]-5), (max_act[0]-5, max_act[1]-4), (max_act[0]-5, max_act[1]-3), (max_act[0]-5, max_act[1]-2), (max_act[0]-5, max_act[1]-1), (max_act[0]-5, max_act[1]), (max_act[0]-5, max_act[1]+1), (max_act[0]-5, max_act[1]+2), (max_act[0]-5, max_act[1]+3), (max_act[0]-5, max_act[1]+4), (max_act[0]-5, max_act[1]+5),
           (max_act[0]-4, max_act[1]-5), (max_act[0]-4, max_act[1]-4), (max_act[0]-4, max_act[1]-3), (max_act[0]-4, max_act[1]-2), (max_act[0]-4, max_act[1]-1), (max_act[0]-4, max_act[1]), (max_act[0]-4, max_act[1]+1), (max_act[0]-4, max_act[1]+2), (max_act[0]-4, max_act[1]+3), (max_act[0]-4, max_act[1]+4), (max_act[0]-4, max_act[1]+5),           
           (max_act[0]-3, max_act[1]-5), (max_act[0]-3, max_act[1]-4), (max_act[0]-3, max_act[1]-3), (max_act[0]-3, max_act[1]-2), (max_act[0]-3, max_act[1]-1), (max_act[0]-3, max_act[1]), (max_act[0]-3, max_act[1]+1), (max_act[0]-3, max_act[1]+2), (max_act[0]-3, max_act[1]+3), (max_act[0]-3, max_act[1]+4), (max_act[0]-3, max_act[1]+5),
           (max_act[0]-2, max_act[1]-5), (max_act[0]-2, max_act[1]-4), (max_act[0]-2, max_act[1]-3), (max_act[0]-2, max_act[1]-2), (max_act[0]-2, max_act[1]-1), (max_act[0]-2, max_act[1]), (max_act[0]-2, max_act[1]+1), (max_act[0]-2, max_act[1]+2), (max_act[0]-2, max_act[1]+3), (max_act[0]-2, max_act[1]+4), (max_act[0]-2, max_act[1]+5),
           (max_act[0]-1, max_act[1]-5), (max_act[0]-1, max_act[1]-4), (max_act[0]-1, max_act[1]-3), (max_act[0]-1, max_act[1]-2), (max_act[0]-1, max_act[1]-1), (max_act[0]-1, max_act[1]), (max_act[0]-1, max_act[1]+1), (max_act[0]-1, max_act[1]+2), (max_act[0]-1, max_act[1]+3), (max_act[0]-1, max_act[1]+4), (max_act[0]-1, max_act[1]+5),
           (max_act[0], max_act[1]-5), (max_act[0], max_act[1]-4), (max_act[0], max_act[1]-3), (max_act[0], max_act[1]-2), (max_act[0], max_act[1]-1), (max_act[0], max_act[1]), (max_act[0], max_act[1]+1), (max_act[0], max_act[1]+2), (max_act[0], max_act[1]+3), (max_act[0], max_act[1]+4), (max_act[0], max_act[1]+5),
           (max_act[0]+1, max_act[1]-5), (max_act[0]+1, max_act[1]+1), (max_act[0]+1, max_act[1]-3), (max_act[0]+1, max_act[1]-2), (max_act[0]+1, max_act[1]-1), (max_act[0]+1, max_act[1]), (max_act[0]+1, max_act[1]+1), (max_act[0]+1, max_act[1]+2), (max_act[0]+1, max_act[1]+3), (max_act[0]+1, max_act[1]+4), (max_act[0]+1, max_act[1]+5),
           (max_act[0]+2, max_act[1]-5), (max_act[0]+2, max_act[1]+2), (max_act[0]+2, max_act[1]-3), (max_act[0]+2, max_act[1]-2), (max_act[0]+2, max_act[1]-1), (max_act[0]+2, max_act[1]), (max_act[0]+2, max_act[1]+1), (max_act[0]+2, max_act[1]+2), (max_act[0]+2, max_act[1]+3), (max_act[0]+2, max_act[1]+4), (max_act[0]+2, max_act[1]+5),
           (max_act[0]+3, max_act[1]-5), (max_act[0]+3, max_act[1]+3), (max_act[0]+3, max_act[1]-3), (max_act[0]+3, max_act[1]-2), (max_act[0]+3, max_act[1]-1), (max_act[0]+3, max_act[1]), (max_act[0]+3, max_act[1]+1), (max_act[0]+3, max_act[1]+2), (max_act[0]+3, max_act[1]+3), (max_act[0]+3, max_act[1]+4), (max_act[0]+3, max_act[1]+5),
           (max_act[0]+4, max_act[1]-5), (max_act[0]+4, max_act[1]+4), (max_act[0]+4, max_act[1]-3), (max_act[0]+4, max_act[1]-2), (max_act[0]+4, max_act[1]-1), (max_act[0]+4, max_act[1]), (max_act[0]+4, max_act[1]+1), (max_act[0]+4, max_act[1]+2), (max_act[0]+4, max_act[1]+3), (max_act[0]+4, max_act[1]+4), (max_act[0]+4, max_act[1]+5),
           (max_act[0]+5, max_act[1]-5), (max_act[0]+5, max_act[1]+5), (max_act[0]+5, max_act[1]-3), (max_act[0]+5, max_act[1]-2), (max_act[0]+5, max_act[1]-1), (max_act[0]+5, max_act[1]), (max_act[0]+5, max_act[1]+1), (max_act[0]+5, max_act[1]+2), (max_act[0]+5, max_act[1]+3), (max_act[0]+5, max_act[1]+4), (max_act[0]+5, max_act[1]+5)]
    return act


# In[357]:


def get_patch(layer_name,x):
    
    img = np.expand_dims(x,axis=0)
    feature = get_feature(layer_name,img)
    
    max_act,max_ch = get_max_act(feature)
    
    max_act_val = np.max(np.squeeze(last_conv_feature[:,:,:,max_ch]))
    max_act = np.where(np.squeeze(last_conv_feature[:,:,:,max_ch]) == max_act_val)
    max_act = (max_act[0][0],max_act[1][0])
    
    patch1 = trace_patch1(max_act)
    patch2, patch3, patch4 = [], [], []
    for i in patch1:
        patch2.extend(trace_patch1(i))
    patch2 = list(set(patch2))
    
    for i in patch2:
        patch3.extend(trace_poolpatch1(i))
    patch3 = list(set(patch3))
    
    for i in patch3:
        patch4.extend(trace_patch2(i))
    patch4 = list(set(patch4))
    
    rows, cols = zip(*patch4)
    x1, y1 = np.min(rows), np.min(cols)
    x2, y2 = np.max(rows), np.max(cols)
    roi = x[x1:x2+1, y1:y2+1]
    return roi


# In[358]:


for i in range(5):
    
    p = get_patch('conv_layer2',train_x[50*i])
    fig, ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].imshow(train_x[50*i])
    ax[0].axis('off')
    ax[1].imshow(p)
    ax[1].axis('off')
    plt.savefig('./models/CNN1/patch_'+str(50*i)+'.png')
    plt.show()
    plt.close()


# In[ ]:




