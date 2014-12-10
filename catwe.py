import numpy as np  # a conventional alias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
import os,sys,glob



filenames = []
rootdir = sys.argv[1]
startday = sys.argv[2]
for subdir, dirs, files in os.walk(rootdir):
	if '-' not in subdir:
		continue
	if os.path.isfile(subdir+'/out.data'):
		filenames.append(subdir+'/out.data')

def date_compare(x, y):
		if int(x.split('/')[4].split('-')[1]) == int(y.split('/')[4].split('-')[1]):
			return int(x.split('/')[4].split('-')[2]) - int(y.split('/')[4].split('-')[2])
		return int(x.split('/')[4].split('-')[1]) - int(y.split('/')[4].split('-')[1])

filenames.sort(date_compare)
i = 0
iwd = 0
trainingfiles =[]
weekend = 0
holiday = 0
for f in filenames[int(startday)-1:]:
	i = i + 1
	if weekend > 0:
		weekend = weekend - 1
		continue
	iwd = iwd + 1
	trainingfiles.append(f)
	if iwd % 5 == 0:
		trainingfiles[-1] = f+'.wk'
		outfile = open(f+'.wk', 'wb')
		for j in range(3):
			if int(startday)+i-1+j > 364:
				continue
			with open(filenames[int(startday)+i-1+j], 'rb') as readfile:
				shutil.copyfileobj(readfile, outfile)
		weekend = 2
		continue
		'../Data/2010/1/2010-1-18/out.data'
def remove_holiday(f):
	try:
		ind = trainingfiles.index(f)
	except:
		ind = trainingfiles.index(f+'.wk')
	if ind < 1:
		return
	item = trainingfiles[ind-1]
	trainingfiles.remove(item)
	return

remove_holiday('../Data/2010/1/2010-1-18/out.data')
remove_holiday('../Data/2010/2/2010-2-15/out.data')
remove_holiday('../Data/2010/4/2010-4-2/out.data')
remove_holiday('../Data/2010/5/2010-5-31/out.data')
remove_holiday('../Data/2010/7/2010-7-5/out.data')
remove_holiday('../Data/2010/9/2010-9-6/out.data')
remove_holiday('../Data/2010/11/2010-11-25/out.data')
remove_holiday('../Data/2010/12/2010-12-24/out.data')

