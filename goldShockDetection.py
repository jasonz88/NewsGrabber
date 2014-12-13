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

import numpy as np  # a conventional alias
import shutil
import os,sys,glob
from math import sqrt

#filenames = tuple(open(sys.argv[1],'r'))
ins = open( sys.argv[1], "r" )
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
print len(jumps)

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

print minNorm
print maxNorm

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
for line in yTotalDataNormed:
	curNorm = sqrt(sum(s**2 for s in line))
	if curNorm > 0.10:
		shockDays.append(line)
		ind = lineNumber
		sample = linesInFile[ind+1]
		year = sample.split(' ')[2].split('\t')[0]
		month = str(dictMonth[sample.split(' ')[0]])
		day = str(int(sample.split(' ')[1][:-1]))
		if int(year) > 2009 and int(year) < 2015:
			shockDaysWithNews.append(line)
			inds = filenames.index(Datadir+'/'+year+'/'+month+'/'+year+'-'+month+'-'+day+'/out.data')
			shockDaysNews.append(filenames[inds]+'.sk')
			outfile = open(shockDaysNews[-1], 'wb')
			for i in range(-2,3):
				if inds + i < 0 or inds + i > len(filenames) - 1:
					continue
				with open(filenames[inds+i], 'rb') as readfile:
					shutil.copyfileobj(readfile, outfile)
	lineNumber = lineNumber + 1

print len(shockDays)
print len(shockDaysNews)
print len(shockDaysWithNews)




# print(yTotalDataNormed)
# sys.exit(1)


# km.fit(yTotalDataNormed[::-1])
# y_total = km.labels_
# cc = km.cluster_centers_

# y_train = y_total[486:]
# y_test = y_total[234:486]





# filenames = []



# rootdir = sys.argv[1]
# startday = 4
# for subdir, dirs, files in os.walk(rootdir):
# 	if '-' not in subdir:
# 		continue
# 	if os.path.isfile(subdir+'/out.data'):
# 		filenames.append(subdir+'/out.data')

# def date_compare(x, y):
# 		if int(x.split('/')[4].split('-')[1]) == int(y.split('/')[4].split('-')[1]):
# 			return int(x.split('/')[4].split('-')[2]) - int(y.split('/')[4].split('-')[2])
# 		return int(x.split('/')[4].split('-')[1]) - int(y.split('/')[4].split('-')[1])

# filenames.sort(date_compare)
# i = 0
# iwd = 0
# trainingfiles =[]
# weekend = 0
# holiday = 0
# for f in filenames[int(startday)-1:]:
# 	i = i + 1
# 	if weekend > 0:
# 		weekend = weekend - 1
# 		continue
# 	iwd = iwd + 1
# 	trainingfiles.append(f)
# 	if iwd % 5 == 0:
# 		weekend = 2
# 		continue
# 	if iwd % 5 == 1:
# 		trainingfiles[-1] = f+'.wk'
# 		outfile = open(f+'.wk', 'wb')
# 		for j in range(3):
# 			if int(startday)+i-2-j < 0:
# 				continue
# 			with open(filenames[int(startday)+i-2-j], 'rb') as readfile:
# 				shutil.copyfileobj(readfile, outfile)

# def remove_holiday(f):
# 	try:
# 		ind = trainingfiles.index(f)
# 	except:
# 		if f+'.wk' in trainingfiles:
# 			ind = trainingfiles.index(f+'.wk')
# 		else:
# 			return
# 	if ind < 0:
# 		return
# 	item = trainingfiles[ind]
# 	trainingfiles.remove(item)
# 	return


# remove_holiday('../Data/2010/1/2010-1-18/out.data')
# remove_holiday('../Data/2010/2/2010-2-15/out.data')
# remove_holiday('../Data/2010/4/2010-4-2/out.data')
# remove_holiday('../Data/2010/5/2010-5-31/out.data')
# remove_holiday('../Data/2010/7/2010-7-5/out.data')
# remove_holiday('../Data/2010/9/2010-9-6/out.data')
# remove_holiday('../Data/2010/11/2010-11-25/out.data')
# remove_holiday('../Data/2010/12/2010-12-24/out.data')

# for f in trainingfiles:
# 	print f



