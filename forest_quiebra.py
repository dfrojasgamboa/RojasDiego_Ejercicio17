import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
from scipy.io import arff # para leer archivos .arff


files = []
for f in range(5):
    file = pd.DataFrame(arff.loadarff( str(f+1) + 'year.arff')[0])
    files.append( file )
    
data = pd.concat(files, axis=0, join='outer')
data = data.dropna()

purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['class']==b'0')
purchasebin[ii] = 0

data['class'] = purchasebin
target = data['class']

data = data.drop(['class'],axis=1)
labels = pd.read_csv( 'Attr.csv' )
labels.keys()
predictors = labels['label']


X_train, X_, y_train, y_ = sklearn.model_selection.train_test_split(data, target, test_size=0.5)
X_valid, X_test, y_valid, y_test = sklearn.model_selection.train_test_split(X_, y_, test_size=0.6)

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_features='sqrt')


n_trees = np.arange(1,10,1)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), len(predictors)))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test)))
    feature_importance[i, :] = clf.feature_importances_
    
    
    
plt.figure()    
plt.scatter(n_trees, f1_test)
plt.savefig('features.png')

M = n_trees[np.argmax(f1_test)]
F = f1_test[np.argmax(f1_test)]

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=M, max_features='sqrt')
clf.fit(X_valid, y_valid)
feature_importance = clf.feature_importances_


# Grafica los features mas importantes
a = pd.Series(feature_importance, index=predictors)

plt.figure()
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.savefig( str(M) + 'arboles, F1-score=' + str(F)+'.png')