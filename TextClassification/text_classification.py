# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:30:35 2017

@author: Andreu Mora
"""

f = open('trainingdata.txt')
# Load the file
N = [int(x) for x in f.readline().strip().split()]
N = N[0]
train_text = N*['']
train_cat = N*[0]
for i in range(N):
    loaded_string = f.readline().strip().split()
    train_text[i] = ' '.join(loaded_string[1:])
    train_cat[i] = int(loaded_string[0])
f.close()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

vec_count = CountVectorizer()
train_x = vec_count.fit_transform(train_text)
tf_idf = TfidfTransformer(use_idf=False).fit(train_x)
train_x = tf_idf.transform(train_x)
clf = MultinomialNB()
clf = clf.fit(train_x, train_cat)
predicted = clf.predict(train_x)
print classification_report(train_cat, predicted)

# Results aren't appauling, use an SVM
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(alpha=1e-3)
clf = clf.fit(train_x, train_cat)
predicted = clf.predict(train_x)
print classification_report(train_cat, predicted)

# SVM is much much better, use GridSearch to optimize results
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
text_clf = Pipeline ([('vec', CountVectorizer()), \
                      ('tf_idf', TfidfTransformer()), \
                      ('clf', SGDClassifier())])
#text_clf = text_clf.fit(train_text, train_cat)
params_to_tweak = {'tf_idf__use_idf':(True, False), 'clf__alpha':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
gs_clf = GridSearchCV(text_clf, params_to_tweak, n_jobs=1)
gs_clf = gs_clf.fit(train_text, train_cat)
predicted = gs_clf.predict(train_text)
print classification_report(train_cat, predicted)
for param in params_to_tweak:
    print '{:s} => {:s}'.format(param, str(gs_clf.best_params_[param]))




