import numpy as np  # a conventional alias
import shutil
import os,sys,glob



filenames = []
rootdir = sys.argv[1]
startday = 2
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
iwd = 3
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
		weekend = 2
		continue
	if iwd % 5 == 1:
		trainingfiles[-1] = f+'.wk'
		outfile = open(f+'.wk', 'wb')
		for j in range(3):
			if int(startday)+i-2-j < 0:
				continue
			with open(filenames[int(startday)+i-2-j], 'rb') as readfile:
				shutil.copyfileobj(readfile, outfile)

def remove_holiday(f):
	try:
		ind = trainingfiles.index(f)
	except:
		if f+'.wk' in trainingfiles:
			ind = trainingfiles.index(f+'.wk')
		else:
			return
	if ind < 0:
		return
	item = trainingfiles[ind]
	trainingfiles.remove(item)
	return

remove_holiday('../Data/2014/1/2014-1-20/out.data')
remove_holiday('../Data/2014/2/2014-2-17/out.data')
remove_holiday('../Data/2014/4/2014-4-18/out.data')
remove_holiday('../Data/2014/5/2014-5-26/out.data')
remove_holiday('../Data/2014/7/2014-7-4/out.data')
remove_holiday('../Data/2014/9/2014-9-1/out.data')
remove_holiday('../Data/2014/11/2014-11-16/out.data')
remove_holiday('../Data/2014/11/2014-11-23/out.data')
remove_holiday('../Data/2014/11/2014-11-27/out.data')
remove_holiday('../Data/2014/11/2014-11-30/out.data')
remove_holiday('../Data/2014/12/2014-12-5/out.data')
remove_holiday('../Data/2014/12/2014-12-6/out.data')
remove_holiday('../Data/2014/12/2014-12-7/out.data')
remove_holiday('../Data/2014/12/2014-12-8/out.data')
remove_holiday('../Data/2014/12/2014-12-9/out.data')


for f in trainingfiles:
	print f



