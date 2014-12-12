import numpy as np  # a conventional alias
import shutil
import os,sys,glob



filenames = []
rootdir = sys.argv[1]
startday = 3
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

remove_holiday('../Data/2011/1/2011-1-17/out.data')
remove_holiday('../Data/2011/2/2011-2-21/out.data')
remove_holiday('../Data/2011/4/2011-4-22/out.data')
remove_holiday('../Data/2011/5/2011-5-30/out.data')
remove_holiday('../Data/2011/7/2011-7-4/out.data')
remove_holiday('../Data/2011/9/2011-9-5/out.data')
remove_holiday('../Data/2011/11/2011-11-24/out.data')
remove_holiday('../Data/2011/12/2011-12-26/out.data')

for f in trainingfiles:
	print f



