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
iwd = 2
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

remove_holiday('../Data/2013/1/2013-1-21/out.data')
remove_holiday('../Data/2013/2/2013-2-18/out.data')
remove_holiday('../Data/2013/3/2013-3-29/out.data')
remove_holiday('../Data/2013/5/2013-5-27/out.data')
remove_holiday('../Data/2013/7/2013-7-4/out.data')
remove_holiday('../Data/2013/9/2013-9-2/out.data')
remove_holiday('../Data/2013/11/2013-11-28/out.data')
remove_holiday('../Data/2013/12/2013-12-25/out.data')

for f in trainingfiles:
	print f



