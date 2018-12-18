# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:26:07 2018

@author: Mikko Impiö
"""

import json
import os
import numpy as np
import pandas as pd
from utils import find_similar

food_mappings = np.load('food_mappings.npy').item()
meta_path = 'C:\\koodia\\huawei\\Yummly28K\\metadata27638'
servings = {}
ingredients = {}

for label in food_mappings:
    r_ind = food_mappings[label][1] +1 #+1 because filenames start with 1
    if r_ind != 0:
        f_str = str(r_ind).zfill(5)
        jsonpath = os.path.join(meta_path, 'meta' + f_str + '.json')
        json_data=open(jsonpath, encoding='utf8').read()
        data = json.loads(json_data)
        
        serves = data["numberOfServings"]
        servings[label] = serves
        
        ing = data["ingredientLines"]
        ingredients[label] = ing
    else:
        servings[label] = -1
        ingredients[label] = ['nan']

l = []
for key in ingredients:
    for ing in ingredients[key]:
        l.append(ing)
    
l = list(set(l))
I = pd.Series(l)

I = I.replace("\d|\/|[()%-]|[¼⅔½⅓¾]", "",regex=True)
words = ["cups ",
         "cup ",
         "Cups ",
         "Cup ",
         "tbsps",
         "tbsp",
         "tsp ",
         "tablespoons ",
         "tablespoon ",
         "teaspoons ",
         "teaspoon ",
         "ounces ",
         "ounce ",
         "oz ",
         "pounds ",
         "pound ",
         "lb ",
         "g ",
         "pkg ",
         "or ",
         "of ",
         "⁄ .",
         "\.",
         "\,"]

for word in words:
    I = I.replace(word,"",regex=True)
    
I = I.apply(lambda x: x.strip())
I = list(set(I))


co2data = pd.read_csv('C:\\Users\\Mikko Impiö\\Google Drive\\koodia\\huawei\\CO2_classifier\\co2_data.csv', sep=";")

## FIND SIMILAR
co2list = co2data.name.values.flatten().tolist()
diffs, similars, sim_indices = find_similar(l, co2list)

#sim = {}
#for key in similars:
#    li = str(similars[key])
#    sim[key] = li
#sim = pd.DataFrame.from_dict(sim, orient='index')


#sim.to_csv('similars.csv',sep=";",encoding='utf8',header=None)

sim_id = pd.read_excel('C:\\Users\\Mikko Impiö\\Google Drive\\koodia\\huawei\\CO2_classifier\\similars_excel_with_id.xlsx', sep=";")
df = sim_id.set_index("id")
df.ingredient = sim.index

df = df.drop(df[df.co2id == 0].index)


cheese = df[df.ingredient.str.contains("cheese",case=False)] 
milk = df[df.ingredient.str.contains("milk",case=False)] 
soy_milk = df[df.ingredient.str.contains("soy milk",case=False)] 
eggs = pd.concat([df[df.ingredient.str.contains("egg",case=False)],
                   df[df.ingredient.str.contains("eggs",case=False)]
                   ])  
garlic = df[df.ingredient.str.contains("garlic",case=False)]
butter = df[df.ingredient.str.contains("butter",case=False)]
tomato = df[df.ingredient.str.contains("tomato",case=False)]
onion = df[df.ingredient.str.contains("onion",case=False)]
cream = df[df.ingredient.str.contains("cream",case=False)]
chicken = df[df.ingredient.str.contains("chicken",case=False)]
potato = df[df.ingredient.str.contains("potato",case=False)]
pepper = df[df.ingredient.str.contains("pepper",case=False)]
carrot = df[df.ingredient.str.contains("carrot",case=False)]
beef = df[df.ingredient.str.contains("beef",case=False)]
squid = df[df.ingredient.str.contains("squid",case=False)]
pork = df[df.ingredient.str.contains("pork",case=False)]
cucumber = df[df.ingredient.str.contains("cucumber",case=False)]

df.loc[cheese.index, "co2id"] = 127
df.loc[milk.index, "co2id"] = 79
df.loc[soy_milk.index, "co2id"] = 60
df.loc[eggs.index, "co2id"] = 103
df.loc[garlic.index, "co2id"] = 56
df.loc[butter.index, "co2id"] = 128
df.loc[tomato.index, "co2id"] = 62
df.loc[onion.index, "co2id"] = 1
df.loc[cream.index, "co2id"] = 114

df.loc[chicken.index, "co2id"] = 108
df.loc[potato.index, "co2id"] = 3
df.loc[pepper.index, "co2id"] = 59
df.loc[carrot.index, "co2id"] = 4
df.loc[beef.index, "co2id"] = 139
df.loc[squid.index, "co2id"] = 120
df.loc[pork.index, "co2id"] = 115
df.loc[cucumber.index, "co2id"] = 95


df = df.drop("similars",axis=1)

## SAVE TO CSV AND MANUALLY FILL

filled = pd.read_csv('C:\\Users\\Mikko Impiö\\Google Drive\\koodia\\huawei\\CO2_classifier\\manual_fill.csv', sep=";",encoding='utf8')
filled = filled.set_index("id")
df.loc[filled.index, "co2id"] = filled.co2id


fullmap = sim_id.set_index("id")
fullmap.loc[df.index, "co2id"] = df.co2id
fullmap = fullmap.drop("similars",axis=1)


#np.save('co2map.npy',fullmap)
