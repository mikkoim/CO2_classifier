# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:42:15 2018

@author: Mikko Impiö
"""
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import random


def find_all_files(path):
    
    filenames = []
    
    for root, dirs, files in os.walk(path):
        for name in files:
            
            fullname = os.path.join(root, name)
            if fullname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):        
                filenames.append(fullname)
                
    return filenames

def files_from_json(imgroot, jsonpath):
    df = pd.read_json(jsonpath)
    fnamelist = []
    for col in df.columns:
        fnames = df[col].apply(lambda x: (imgroot + os.sep + x + ".jpg").replace("/", os.sep))
        fnamelist.append(fnames)
        
    filenames = pd.concat(fnamelist).tolist()
    return filenames


def get_label_binarizer(path):
    
    label_names = pd.read_csv(path, header=None)
    label_names = label_names.values.flatten().tolist()

    lb = LabelBinarizer()
    lb.fit(label_names)
    
    return lb

def get_label_encoder(path):
    
    label_names = pd.read_csv(path, header=None)
    label_names = label_names.values.flatten().tolist()

    lb = LabelEncoder()
    lb.fit(label_names)
    
    return lb
    
def data_generator(batch_size, imsize, path, labelpath, json=False, jsonpath = None, israndom= False):
    
    if json == False:
        files = find_all_files(path)
    else:
        files = files_from_json(path, jsonpath)
                
    lb = get_label_binarizer(labelpath)
    
    print("Found {} images".format(len(files)))
    
    j = 0    
    while True:
        inputs = []
        targets = []
#        print("j: {}\nlenfiles: {}".format(j,len(files)))
        for i in range(batch_size):
            
            if israndom:
                name = random.choice(files)
            else:
                name = files[j+i]
            
            image = imread(name)
            image = resize(image, imsize[0:2]).astype(float)
            image = image - np.min(image)
            image = image / np.max(image)
    #        image = 255 * image / np.max(image)
    #        image = np.array(image.astype(np.uint8))
    #        
    #        image = image.astype(float) / 255.0
            
            # Extract class label
            c = name.split(os.sep)[-2]
            label = lb.transform([c])[0]
            
            inputs.append(image)
            targets.append(label)
        j = j + i
        yield np.asarray(inputs), np.asarray(targets)
        
        
def load_data(path, jsonpath, imsize, percentage):
    
    files = files_from_json(path, jsonpath)
    
    np.random.shuffle(files)
    
    X = []
    y = []

    print("Found {} images".format(len(files)))
    
    n_labels = np.uint32(np.round(percentage*len(files)))
    i = 1
    for name in files[0:n_labels]:
        image = imread(name)
        image = resize(image, imsize[0:2]).astype(float)
        image = image - np.min(image)
        image = 255 * image / np.max(image)
        X.append(image.astype(np.uint8))
        
        # Extract class label
        c = name.split(os.sep)[-2]
        y.append(c)
        
        print("{}/{} read".format(i,n_labels))
        i += 1

    X = np.array(X)
    y = np.array(y)
    
    return X, y


def load_old_model(filename):
    from keras.utils.generic_utils import CustomObjectScope
    import keras

    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = keras.models.load_model(filename)
    return model


def find_similar(a,b,k=10):
    from difflib import SequenceMatcher
    def diff(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    diffs = {}
    for value in a:
        diffs[value] = []
        for i in range(len(b)):
            dif = diff(b[i], value)
            diffs[value].append(dif)
            print("{}: {}".format(value, i))
    
    similars = {}
    sim_indices = {}
    for key in diffs.keys():
        top_k = np.asarray(diffs[key]).argsort(axis=0)[-k::][::-1]
        recipe = np.asarray(b)[top_k]
        similars[key] = recipe
        sim_indices[key] = top_k
        
    return diffs, similars, sim_indices
