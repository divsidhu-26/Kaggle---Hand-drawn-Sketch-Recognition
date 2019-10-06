
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend
from keras.utils.np_utils import to_categorical as ctg
import numpy as np
import os,glob,csv
labels = {}
from keras.models import load_model

def read_train_file():
    i = 0
    x,y = [],[]
    for file_name in glob.glob('./train/'+'*.npy'):
        name = file_name.split('/')[2].split('.')[0]
        dum = np.load(file_name)
        x.append(dum/255)
        y.append(i)
        labels[i] = name
        i += 1
    return np.array(x),np.array(y)

def write_csv(data):
    with open("pred_d3_2.csv",'w+') as file:
        file.write("ID,CATEGORY\n")
        for i in range(len(data)):
            file.write(str(i)+","+labels[data[i]]+"\n")
        file.close()


# In[2]:


train_X,train_Y = [],[]
train_X,train_Y = read_train_file()
test_X = np.load("./test/test.npy")/255
std = np.std(test_X,axis=0)
mean = np.mean(test_X,axis=0)
for i in range(len(test_X)):
    test_X[i] = np.divide(np.subtract(test_X[i],mean),std)
all_x = train_X[0][:4500]
all_y = [0 for i in range(4500)]
for i in range(1,len(train_X)):
    all_x = np.concatenate((all_x,train_X[i][:4500]),0)
    dum = [i for j in range(4500)]
    all_y.extend(dum)
valid_y = [0 for i in range(500)]
valid_x = train_X[0][4500:]
for i in range(1,len(train_X)):
    valid_x = np.concatenate((valid_x,train_X[i][4500:]),0)
    dum = [i for j in range(500)]
    valid_y.extend(dum)
std = np.std(all_x,axis=0)
# print(std)
mean = np.mean(all_x,axis=0)
# print(mean)
# print(all_x[0])
for i in range(len(all_x)):
    all_x[i] = np.divide(np.subtract(all_x[i],mean),std)
std = np.std(valid_x,axis=0)
# print(std)
mean = np.mean(valid_x,axis=0)
# print(mean)
# print(all_x[0])
for i in range(len(valid_x)):
    valid_x[i] = np.divide(np.subtract(valid_x[i],mean),std)


# In[3]:


valid_y = ctg(valid_y,num_classes = 20)
all_y = ctg(all_y,num_classes = 20)

all_x = all_x.reshape(len(all_x), 28,28,1)
valid_x = valid_x.reshape(len(valid_x), 28,28,1)
test_X = test_X.reshape(len(test_X), 28,28,1)


# In[ ]:


model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(Convolution2D(64, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(128,kernel_size=(3,3), input_shape = (28,28,1),activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = model.fit(all_x, all_y,
                    batch_size=250,
                    epochs=150,
                    verbose=1,
                    validation_data=(valid_x, valid_y))


# In[8]:


pred = model.predict(test_X,batch_size=128)
test_labels = np.argmax(pred,axis=1)


# In[9]:


write_csv(test_labels)

model.save('my_model.h5')
