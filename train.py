# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 16:21:28 2018

@author: Mikko Impiö
"""

import numpy as np # Numerics


from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import LabelBinarizer

from utils import data_generator, load_old_model


def get_model(imsize, classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = imsize))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(classes, activation = 'sigmoid'))
#    model.add(Dense(len(class_names), activation = 'sigmoid'))
    #####################################
    
    model.summary()
    return model


def get_pretrained_model(imsize, classes, name):
    
    if name == 'MobileNet':
        base_model = MobileNet(input_shape = imsize, alpha = 0.25, include_top = False)
        z = base_model.outputs[0]
        z = Flatten()(z)
        z = Dense(classes, activation = 'softmax')(z)
        model = Model(inputs = base_model.inputs, output = z)
        
    elif name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(256, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


if __name__ == "__main__":
    datapath = "C:\\koodia\\huawei\\food-101\\food-101\\images"
    labelpath = 'C:\\koodia\\huawei\\food-101\\food-101\\meta\\classes.txt'
    
    trainpath = "C:\\koodia\\huawei\\food-101\\food-101\\meta\\train.json"
    testpath =  "C:\\koodia\\huawei\\food-101\\food-101\\meta\\test.json"
    
    imsize = (128,128,3)
    
    dgtrain = data_generator(64, imsize, datapath, labelpath, json=True, jsonpath = trainpath, israndom=True) 
    dgtest =  data_generator(64, imsize, datapath, labelpath, json=True, jsonpath = testpath, israndom = True) 
    
#    model = get_model(imsize, 101)
    model = get_pretrained_model(imsize, 101, 'MobileNet')
#    model = load_old_model('mobilenet_generator_128_240epochs_model.h5')
    
    model.summary()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    ## If true trains with data loaded with a generator, 
    ## if false, loads data to memory and trains with this
    if False:
        num_steps = 100
        num_epochs = 100
        
        model.fit_generator(dgtrain, 
                            validation_data= dgtest, 
                            steps_per_epoch= num_steps, 
                            epochs = num_epochs,
                            validation_steps = num_steps)
    
    ## muistiin ladatulla datalla koulutus
    else:
#        X, y = load_data(datapath, trainpath, imsize, 0.5)
        X = np.load('C:\koodia\huawei\X_128_half_random.npy')
        y = np.load('C:\koodia\huawei\y_128_half_random.npy')
        
        X = X / 255.0
        
        lb = LabelBinarizer()
        y = lb.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2,
                                                        random_state=0)
        
        model.fit(X_train, y_train, epochs = num_epochs, validation_data = (X_test, y_test))
        
    
    
