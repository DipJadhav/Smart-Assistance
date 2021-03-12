from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
from sklearn import tree
import os
#import graphviz
import sklearn.metrics as m

from sklearn import tree

import numpy as np
import matplotlib.pyplot as plt

base = os.getcwd();
filename_train = 'test.csv'

path = base + '/' + filename_train
data = pd.read_csv(path)
from sklearn.model_selection import train_test_split




train, test = train_test_split(data, test_size=0.7)

feature_cols = ['Duration','Start station number','End station number']
#print(data)
X = train.loc[:,feature_cols]
Y = train.Member


gnb_n = tree.DecisionTreeClassifier(criterion="entropy")
gnb_n.fit(X,Y)





ans_n = gnb_n.predict(test)

print("\n\n\n\n")
print(classification_report(Y,ans_n))
#Precision = TP/(TP+FP)
#Recall = TP/(TP+FN)
#F1 = 2*(recall + precision)/(recall + precision)


#print(prob)
acc_n = m.accuracy_score(ans_n,Y)
a_n = confusion_matrix(Y,ans_n)
#print(ans)
print("\n\n\n\n")
print("Accuracy Of Deciosion Tree : - ",acc_n)
#print(acc)
print("ConfusionMatrix  Of Decision Tree : - \n",a_n)
#print(a)

#print(acc_n)
#print(a_n)
print("===================================================================================================\n\n\n")
