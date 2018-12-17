# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 23:36:28 2018

@author: Mikko Impiö
"""
from utils import load_old_model, load_data, get_label_encoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def get_top_k(y, k):
    top_k = y.argsort(axis=1)[:,-k::][:,::-1]
    return top_k

def make_title(pred, probs):
    title = "GT: {}\npreds:\n".format(gt)
    for i, l in enumerate(pred):
        title = title + l + ": {:.3f}".format(probs[i]) +  "\n"
    return title


datapath = "C:\\koodia\\huawei\\food-101\\food-101\\images"
labelpath = 'C:\\koodia\\huawei\\food-101\\food-101\\meta\\classes.txt'
testpath =  "C:\\koodia\\huawei\\food-101\\food-101\\meta\\test.json"

#LOAD MODEL
imsize = (128,128,3)
model = load_old_model('mobilenet_generator_128_240epochs_model.h5')
X_test, y_test = load_data(datapath, testpath, imsize, 0.01)
X_test = X_test / 255.0

#PREDICTION
y_prob = model.predict(X_test)

y_pred = np.argmax(y_prob, axis = 1)

top_5 = get_top_k(y_prob,5)


# FROM INDEX TO LABEL
le = get_label_encoder(labelpath)

y_predlabels = le.inverse_transform(y_pred)

y_top5_labels = le.inverse_transform(top_5)

y_top5_probs =  y_prob.copy()
y_top5_probs.sort(axis=1)
y_top5_probs = y_top5_probs[:,-5::][:,::-1]

accuracy = accuracy_score(y_test, y_predlabels)

# PLOTTING
i = np.random.randint(len(y_test))
img = X_test[i,:]
gt = y_test[i]
pred = y_top5_labels[i,:]
prob = y_top5_probs[i,:]
plt.imshow(img)
print(make_title(pred, prob))

