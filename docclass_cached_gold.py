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
from math import sqrt

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
op.add_option("--n_fold",
              action="store", type=int, default=4,
              help="n_fold - 1 fold for training and 1 for testing.")
op.add_option("--shockThreshold",
              action="store", type=float, default=0.10,
              help="minimal norm of price change vector to be considered shock")
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
        'pattern_8',
        'pattern_9',
        'pattern_10',
        'pattern_11',
        'pattern_12',
        'pattern_13',
        'pattern_14',
    ]
    categories = categories[:opts.n_clusters]

# if opts.filtered:
#     remove = ('headers', 'footers', 'quotes')
# else:
#     remove = ()


#filenames = tuple(open(sys.argv[1],'r'))
ins = open('gold-historical-data.txt', "r" )
linesInFile = []
for line in ins:
	linesInFile.append(line[:-1])
ins.close()

jumps = []
i = 3
for line in linesInFile[3:-3]:
	fiveDays = []
	for j in range(5):
		today = linesInFile[i-2+j].split(' ')[2].split('\t')
		yesterday = linesInFile[i-1+j].split(' ')[2].split('\t')
		todayLast = float(today[1])
		todayOpen = float(today[2])
		todayHigh = float(today[3])
		todayLow = float(today[4])
		if today[5] == '-':
			todayVol = float(0)
		else:
			todayVol = float(today[5][:-1])
		todayPer = float(today[6][:-1])
		yesterLast = float(yesterday[1])
		yesterOpen = float(yesterday[2])
		yesterHigh = float(yesterday[3])
		yesterLow = float(yesterday[4])
		if yesterday[5] == '-':
			yesterVol = float(0)
		else:
			yesterVol = float(yesterday[5][:-1])
		yesterPer = float(yesterday[6][:-1])
		fiveDays.append(todayOpen-yesterLast)
		fiveDays.append((todayVol-yesterVol))
		fiveDays.append(todayHigh-todayLow)
		fiveDays.append(todayPer)
	jumps.append(fiveDays)
	i = i + 1

# km = KMeans(n_clusters=8, init='k-means++', max_iter=500, n_init=10,
#                 verbose=0, n_jobs=-1)

# yTotalDataNormed = normalizeInput(yTotalData.yTotalData)
yTotalDataNormed = normalize(jumps, axis=0, norm='l2')

minNorm = 999
maxNorm = -999
for line in yTotalDataNormed:
	curNorm = sqrt(sum(s**2 for s in line))
	if minNorm > curNorm:
		minNorm = curNorm
	if maxNorm < curNorm:
		maxNorm = curNorm

shockDays = []
shockDaysWithNews = []
shockDaysNews = []
dictMonth = {'Dec': 12, 'Nov': 11, 'Oct': 10, 'Sep': 9, 'Aug': 8, 'Jul': 7, 'Jun': 6, 'May': 5, 'Apr': 4, 'Mar': 3, 'Feb': 2, 'Jan': 1}

filenames = []
Datadir = '../Data'
startday = 2
for subdir, dirs, files in os.walk(Datadir):
	if '-' not in subdir:
		continue
	if os.path.isfile(subdir+'/out.data'):
		filenames.append(subdir+'/out.data')

def date_compare(x, y):
		if int(x.split('/')[4].split('-')[1]) == int(y.split('/')[4].split('-')[1]):
			return int(x.split('/')[4].split('-')[2]) - int(y.split('/')[4].split('-')[2])
		return int(x.split('/')[4].split('-')[1]) - int(y.split('/')[4].split('-')[1])

filenames.sort(date_compare)


lineNumber = 0
lookUpTable = {}
l1 = -1
l2 = -1
for line in yTotalDataNormed:
	curNorm = sqrt(sum(s**2 for s in line))
	if curNorm > opts.shockThreshold:
		shockDays.append(line)
		l1 = l1 + 1
		ind = lineNumber
		sample = linesInFile[ind+1]
		year = sample.split(' ')[2].split('\t')[0]
		month = str(dictMonth[sample.split(' ')[0]])
		day = str(int(sample.split(' ')[1][:-1]))
		if int(year) > 2009 and int(year) < 2015:
			shockDaysWithNews.append(line)
			l2 = l2 + 1
			lookUpTable[l2] = l1
			inds = filenames.index(Datadir+'/'+year+'/'+month+'/'+year+'-'+month+'-'+day+'/out.data')
			shockDaysNews.append(filenames[inds]+'.sk')
			outfile = open(shockDaysNews[-1], 'wb')
			for i in range(-2,3):
				if inds + i < 0 or inds + i > len(filenames) - 1:
					continue
				with open(filenames[inds+i], 'rb') as readfile:
					shutil.copyfileobj(readfile, outfile)
	lineNumber = lineNumber + 1

print("shock days: %d, shock days with news: %d" % (len(shockDays), len(shockDaysWithNews)))


km = KMeans(n_clusters=opts.n_clusters, init='k-means++', max_iter=500, n_init=20,
                verbose=0, n_jobs=-1)

km.fit(shockDays)
y_total = km.labels_
cc = km.cluster_centers_

y_train = []
y_test = []

trainingfilenames = []
testfilenames = []

for i in range(len(shockDaysWithNews)):
	if i % opts.n_fold == 0:
		y_test.append(y_total[lookUpTable[i]])
		testfilenames.append(shockDaysNews[i])
	else:
		y_train.append(y_total[lookUpTable[i]])
		trainingfilenames.append(shockDaysNews[i])

print('data loaded training size: %d, testing size: %d' % (len(y_train), len(y_test)))

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
	fd = open('featureNames.txt','r')
	featureNames = pickle.load(fd)
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
	    vectorizer = TfidfVectorizer(input='filename',sublinear_tf=True, max_df=0.8,
	                                 stop_words='english')
	elif opts.use_hashing:
		vectorizer = HashingVectorizer(input='filename',stop_words='english',min_df=15,non_negative=True,
	                                   n_features=opts.n_features)
	else:
		vectorizer = CountVectorizer(input='filename',stop_words='english',min_df=5,max_df=0.8)
	X_train = vectorizer.fit_transform(trainingfilenames)
	duration = time() - t0
	fd = open('X_train.txt','w')
	pickle.dump(X_train,fd)
	fd.close()
	fd = open('featureNames.txt','w')
	featureNames = vectorizer.get_feature_names()
	pickle.dump(featureNames,fd)
	fd.close()

	print("done in %fs" % duration)
	print("n_samples: %d, n_features: %d" % X_train.shape)
	print()

	print("Extracting features from the test dataset using the same vectorizer")
	t0 = time()
	X_test = vectorizer.transform(testfilenames)
	duration = time() - t0
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
    return s if len(s) <= 120 else s[:77] + "..."


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(featureNames)


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