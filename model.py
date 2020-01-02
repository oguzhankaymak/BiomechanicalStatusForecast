# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:47:38 2019

@author: oguzhankaymak
"""

import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("column_3C_weka.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,[-1]].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# # Feature Scaling (Özellik ölçekleme)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# Eğitim veri setine KNN sınıflandırıcısının uygulanması
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

#Accuracy (sınıflandırma için başarım metriği) ölçülmesi
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("Accuracy", acc)
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))