
# coding: utf-8

# In[63]:

from io import BytesIO  
import requests  
import json  
import pandas as pd

def get_data(credentials):  
    """This functions returns a StringIO object containing
    the file content from Bluemix Object Storage V3."""

    url1 = ''.join(['https://identity.open.softlayer.com', '/v3/auth/tokens'])
    data = {'auth': {'identity': {'methods': ['password'],
            'password': {'user': {'name': credentials['username'],'domain': {'id': credentials['domain_id']},
            'password': credentials['password']}}}}}
    headers1 = {'Content-Type': 'application/json'}
    resp1 = requests.post(url=url1, data=json.dumps(data), headers=headers1)
    resp1_body = resp1.json()
    for e1 in resp1_body['token']['catalog']:
        if(e1['type']=='object-store'):
            for e2 in e1['endpoints']:
                        if(e2['interface']=='public'and e2['region']=='dallas'):
                            url2 = ''.join([e2['url'],'/', credentials['container'], '/', credentials['filename']])
    s_subject_token = resp1.headers['x-subject-token']
    headers2 = {'X-Auth-Token': s_subject_token, 'accept': 'application/json'}
    resp2 = requests.get(url=url2, headers=headers2)
    return json.loads(resp2.content)

# @hidden_cell
credentials_1 = {
  'auth_url':'https://identity.open.softlayer.com',
  'project':'object_storage_0969fdd8_3bcd_461d_816e_70b51797f951',
  'project_id':'157e80fbad96454081e560b3fa4261d8',
  'region':'dallas',
  'user_id':'80abdc7f200847ea87a4bfcbf59f443d',
  'domain_id':'7fadf721ad214718a043163b4784bb48',
  'domain_name':'1447597',
  'username':'member_0c09572a4a9762ff605107001c10b0be5bc46b90',
  'password':"""p3sx*K^o7^.t=R0-""",
  'container':'DefaultProjectbhidesuhuskyneuedu',
  'tenantId':'undefined',
  'filename':'data.json'
}

data_json = get_data(credentials_1)
print(data_json.get('no_of_epochs'))


# In[64]:

from __future__ import print_function
import keras
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import pickle
import h5py
import numpy as np
print('Hello')


# In[65]:

# Load label names to use in prediction results
label_list_path = 'datasets/cifar-10-batches-py/batches.meta'

keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
datadir_base = os.path.expanduser(keras_dir)
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
label_list_path = os.path.join(datadir_base, label_list_path)

with open(label_list_path, mode='rb') as f:
    labels = pickle.load(f)

#print(len(labels))


# In[112]:

batch_size = data_json.get('batch_size')
num_classes = 10
epochs = data_json.get('no_of_epochs')
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, shuffled and split between train and test sets:
(x_train_load, y_train_load), (x_test_load, y_test_load) = cifar10.load_data()
print('x_train shape:', x_train_load.shape)
print(x_train_load.shape[0], 'train samples')
print(x_test_load.shape[0], 'test samples')


# In[113]:

# Concatenate the whole data
full_data_x = np.concatenate((x_train_load, x_test_load), axis=0)
full_data_y = np.concatenate((y_train_load, y_test_load), axis=0)
#print('Y full data: ', len(full_data_y))

# Get random images
seed_val = 10
test_data = np.random.randint(0, 60000, size=12000)
x_train = full_data_x[test_data]
y_train = full_data_y[test_data]
#print(len(data_x))
print('y train data: ', y_train.shape[0])

from scipy.stats import itemfreq
count= itemfreq(data_y)
for i in range (0,len(count)):
   print(labels['label_names'][i],count[i][1])


# In[114]:

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape[0])


# In[118]:

class LossAccPerEpochCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[119]:

# Add Calbacks
# checkpoint
outputFolder = './output-callback'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,                              save_best_only=False, save_weights_only=True,                              mode='auto', period=10)
#callbacks_list = [checkpoint]

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,                           verbose=1, mode='auto')

# Custom callback for test data per epoch
custom_callback = LossAccPerEpochCallback((x_train, y_train))

callbacks_list = [earlystop]


# In[120]:

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=data_json.get('learning_rate'), decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_train /= 255

print(x_train.shape[0])
print(y_train.shape[0])
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
'''


# In[121]:

if not data_augmentation:
    print('Not using data augmentation.')
    model_info = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks_list,
              validation_data=(x_train, y_train),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model_info = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        validation_data=(x_train, y_train),
                        workers=4)


# In[122]:

# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate_generator(datagen.flow(x_train, y_train,
                                                   batch_size=batch_size,
                                                   shuffle=False),
                                      steps=x_train.shape[0] // batch_size,
                                      workers=4)
print('Model Accuracy = %.2f' % (evaluation[1]))


# In[123]:

predict_gen = model.predict_generator(datagen.flow(x_train, y_train,
                                                   batch_size=batch_size,
                                                   shuffle=False),
                                      steps=x_train.shape[0] // batch_size,
                                      workers=4)

for predict_index, predicted_y in enumerate(predict_gen):
    actual_label = labels['label_names'][np.argmax(y_train[predict_index])]
    predicted_label = labels['label_names'][np.argmax(predicted_y)]
    print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
                                                          predicted_label))
    if predict_index == num_predictions:
        break


# In[ ]:



