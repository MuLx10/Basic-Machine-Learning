import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
import pandas as pd
#from sklearn import preprocessing, cross_validation, neighbors
style.use('fivethirtyeight')

class KNearestNeighbours(object):
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,raw_data,features,k):
        self.k=k
        nfea=np.array(features)
        if len(np.unique(nfea)) >= self.k:
            warnings.warn('K is set to a value less than total voting groups!')
        self.data={}
        for i in range(len(features)):
            try:
                self.data[features[i]].append(raw_data[i])
            except:
                self.data[features[i]]=[raw_data[i]]
        #print(self.data)
    def predict(self,PREDICT):
        
        VOTES_res=[]
        for pPredict in PREDICT:
            distances = []
            for group in self.data:
                for features in self.data[group]:
                    euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(pPredict))**2))
                    distances.append([euclidean_distance,group])
            #print(sorted(distances)[:k])
            votes = [i[1] for i in sorted(distances)[:self.k]]
            #print(Counter(votes).most_common(2))
            vote_result = Counter(votes).most_common(1)[0][0]
            VOTES_res.append(vote_result)
        return VOTES_res





dataset = [[1,2,6],[2,3,5],[3,1,3],[6,5,2],[7,7,2],[8,6,1],[2,8,1],[1,7,6],[1,9,5]]
features = ['k','k','k','r','r','r','b','b','b']
new_features = [[5,7,6],[2,9,5]]

'''
#[[plt.scatter(j[0],j[1],s=100,color=i) for j in dataset[i]] for i in dataset]
# same as:
for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1],s=100,color=i)
'''
clf=KNearestNeighbours()
clf.fit(dataset,features,5)
result = clf.predict(new_features)
#print(result)
for i in range(len(result)):
    #plt.scatter(new_features[i][0], new_features[i][1], s=100, color = result[i])  
    print(new_features[i],result[i])
#plt.show()


'''
clf2=KNearestNeighbours()
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
print(X_test[0])
clf2.fit(X_train,y_train,2*len(X_train[0])-1)
result = clf.predict(X_test)
'''
