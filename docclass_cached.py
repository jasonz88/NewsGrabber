from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import shutil
import os,sys,glob
import pickle

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--use_tfidf",
              action="store_true",
              help="Use a Tf-Idf vectorizer.")
op.add_option("--n_clusters",
              action="store", type=int, default=8,
              help="n_clusters when using the k-means clustering.")
op.add_option("--n_features",
              action="store", type=int, default=500,
              help="n_features when using the hashing vectorizer.")
op.add_option("--cached_data",
              action="store_true", dest="cached_data",
              help="load cached data")
# op.add_option("--filtered",
              # action="store_true",
              # help="Remove newsgroup information that is easily overfit: "
                   # "headers, signatures, and quoting.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

op.print_help()


###############################################################################






###############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
        'pattern_0',
        'pattern_1',
        'pattern_2',
        'pattern_3',
        'pattern_4',
        'pattern_5',
        'pattern_6',
        'pattern_7',
    ]

# if opts.filtered:
#     remove = ('headers', 'footers', 'quotes')
# else:
#     remove = ()

# print("Loading 20 newsgroups dataset for categories:")
# print(categories if categories else "all")



km = KMeans(n_clusters=opts.n_clusters, init='k-means++', max_iter=500, n_init=1,
                verbose=0, n_jobs=-1)
import yTotalData
def normalizeInput(y):
	n = len(y[0])
	m = len(y)
	for i in range(m):
		y[i][1] = y[i][1] - y[i][0]
		y[i][2] = y[i][2] - y[i][0]
	for j in range(n):
		mSum = 0
		mMin = 9999
		mMax = -9999
		for i in range(m):
			mSum = mSum + y[i][j]
			if mMin > y[i][j]:
				mMin = y[i][j]
			if mMax < y[i][j]:
				mMax = y[i][j]
		mAvg = mSum / m
		mRange = mMax - mMin
		for i in range(m):
			y[i][j] = (y[i][j] - mAvg) / mRange
	return y

# yTotalDataNormed = normalizeInput(yTotalData.yTotalData)
yTotalDataNormed = normalize(yTotalData.yTotalData, axis=0, norm='l1')

# print(yTotalDataNormed)
# sys.exit(1)
km.fit(yTotalDataNormed[::-1])
y_total = km.labels_
cc = km.cluster_centers_

y_train = y_total[486:]
y_test = y_total[234:486]

trainingfilenames = []
testfilenames = []
for i in range(4):
	#filenames = tuple(open(sys.argv[1],'r'))
	ins = open( 'doc201'+str(i)+'.txt', "r" )

	for line in ins:
	    trainingfilenames.append( line[:-1] )
	ins.close()

ins = open( 'doc2014.txt', "r" )

for line in ins:
    testfilenames.append( line[:-1] )
ins.close()

print('data loaded')

# categories = data_train.target_names    # for case categories == None


def size_mb(docs):
    return docs.shape[0] * docs.shape[1] * 8 / 1e6

# split a training set and a test set
# y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training dataset using a sparse vectorizer")
if opts.cached_data:
	fd = open('X_train.txt','r')
	X_train = pickle.load(fd)
	fd.close()
	fd = open('X_test.txt','r')
	X_test = pickle.load(fd)
	fd.close()
	if opts.use_tfidf:
		vectorizer = TfidfVectorizer(input='filename',sublinear_tf=True, max_df=0.5,
	                                 stop_words='english')
	elif opts.use_hashing:
		vectorizer = HashingVectorizer(input='filename',stop_words='english',min_df=15,non_negative=True,
	                                   n_features=opts.n_features)
	else:
		vectorizer = CountVectorizer(input='filename',stop_words='english',min_df=5)
else:
	t0 = time()
	if opts.use_tfidf:
	    vectorizer = TfidfVectorizer(input='filename',sublinear_tf=True, max_df=0.5,
	                                 stop_words='english')
	elif opts.use_hashing:
		vectorizer = HashingVectorizer(input='filename',stop_words='english',min_df=15,non_negative=True,
	                                   n_features=opts.n_features)
	else:
		vectorizer = CountVectorizer(input='filename',stop_words='english',min_df=10)
	X_total = vectorizer.fit_transform(trainingfilenames)
	X_train = X_total[:756]
	X_test = X_total[756:]
	duration = time() - t0
	fd = open('X_train.txt','w')
	pickle.dump(X_train,fd)
	fd.close()

	print("done in %fs" % duration)
	print("n_samples: %d, n_features: %d" % X_train.shape)
	print()

	# print("Extracting features from the test dataset using the same vectorizer")
	# t0 = time()
	# X_test = vectorizer.transform(testfilenames)
	# duration = time() - t0
	fd = open('X_test.txt','w')
	pickle.dump(X_test,fd)
	fd.close()

	print("done in %fs" % duration)
	print("n_samples: %d, n_features: %d" % X_test.shape)
	print()

data_train_size_mb = size_mb(X_train)
data_test_size_mb = size_mb(X_test)

print("%d documents - %0.3fMB (training set)" % (
     X_train.shape[0], data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
     X_test.shape[0], data_test_size_mb))
print("%d categories" % len(categories))
print()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("done in %fs" % (time() - t0))
    print()


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(vectorizer.get_feature_names())


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()