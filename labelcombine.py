# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 21:21:43 2018

@author: Mikko Impiö
"""
import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    # IMAGE CLASSIFICATION LABELS
    labelpath = 'C:\\koodia\\huawei\\food-101\\food-101\\meta\\labels.txt'
    classpath = 'C:\\koodia\\huawei\\food-101\\food-101\\meta\\classes.txt'
    label_names = pd.read_csv(labelpath, header=None)
    label_names = label_names.values.flatten().tolist()
    
    class_names = pd.read_csv(classpath, header=None)
    class_names = class_names.values.flatten().tolist()
    
    
    # RECIPE NAMES
    
    recipepath = "C:\\koodia\\huawei\\Yummly28K\\data_records_27638.txt"
    recipes = pd.read_csv(recipepath, header=None)
    
    recipes = recipes.replace("\d|\t", "",regex=True)
    recipes = recipes.replace("-|_", " ",regex=True)
    
    rem_words = ["My recipes", 
                 "My Recipes", 
                 "Myrecipes",
                 "MyRecipes",
                 
                 "Allrecipes",
                 "AllRecipes",
                 "All recipes",
                 "All Recipes",
                 
                 "Martha Stewart",
                 "Epicurious"]
    
    for word in rem_words:
        recipes = recipes.replace(word, "", regex=True)
    
    ## FIND SIMILAR

    from difflib import SequenceMatcher
    
    def diff(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    recipelist = recipes.values.flatten().tolist()
    
    diffs = {}
    for label in label_names:
        diffs[label] = []
        for i in range(len(recipelist)):
            dif = diff(recipelist[i], label)
            diffs[label].append(dif)
            print("{}: {}".format(label, i))
            
    similars = {}
    for key in diffs.keys():
        top_5 = np.asarray(diffs[key]).argsort(axis=0)[-10::][::-1]
        recipe = np.asarray(recipelist)[top_5]
        similars[key] = recipe
            
