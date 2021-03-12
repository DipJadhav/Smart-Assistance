from sklearn import datasets
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import tree
import os
#import graphviz
import sklearn.metrics as m

base = os.getcwd();
filename_train = 'test.csv'

path = base + '/' + filename_train

#from sklearn import ensemble
#gnb = ensemble.RandomForestClassifier()

from sklearn import tree
gnb = tree.DecisionTreeClassifier()

data = pd.read_csv(path)
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.7)


feature_cols = ['Duration','Start station number','End station number']
#print(data)
X = train.loc[:,feature_cols]
Y = train.Member

#print(Y)
gnb.fit(X,Y)
'''
temp = tree.export_graphviz(gnb,out_file=None,feature_names =feature_cols,class_names=['Casual','Member'],filled=True)  
graph = graphviz.Source(temp)
graph.render("my")
'''
print('test')
#print(test.head())
print(len(test),len(train))
X_test = test.loc[:,feature_cols]
Y_test = test.Member
ans = gnb.predict(X_test)

acc = m.accuracy_score(ans,Y_test)
a = confusion_matrix(Y_test,ans)
print(ans)
print(acc)
print(a)
