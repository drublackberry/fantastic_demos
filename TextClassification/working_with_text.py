# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:04:38 2017

@author: Andreu Mora
"""
import logging
logging.basicConfig()

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# Load the data
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print twenty_train.target_names
print len(twenty_train.data)
print len(twenty_train.filenames)
print "\n".join(twenty_train.data[0].split("\n")[:3])
print twenty_train.target_names[twenty_train.target[0]]

# Extract features
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(twenty_train.data)
print X_train_counts.shape
print count_vec.vocabulary_.get('algorithm')

# From occurrences to frequencies
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)
print X_train_tfidf.shape

# Classify using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Predict new documents
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vec.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print '{:s} => {:s}'.format(doc, twenty_train.target_names[category])
    
# Building a pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Evaluation of the performance
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print np.mean(predicted==twenty_test.target)

# Using an SVM
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline ([('vect', CountVectorizer()), \
                      ('tfidf', TfidfTransformer()), \
                      ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print np.mean(predicted==twenty_test.target)

# Better performance statistics
from sklearn import metrics
print metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names)

# Parameter tuning using grid_search
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1,1), (1,2)], \
              'tfidf__use_idf': (True, False), \
              'clf__alpha': (1e-2, 1e-3) }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
print twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print '{:s} => {:s}'.format(param_name, str(gs_clf.best_params_[param_name]))
