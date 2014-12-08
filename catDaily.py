import shutil
import os,sys,glob
outfilename = sys.argv[1]

rootdir = os.getcwd()
for subdir, dirs, files in os.walk(rootdir):
	print subdir
	if '-' not in subdir:
		continue
	os.chdir(subdir)
	outfile = open(outfilename, 'wb')
	for filename in glob.glob('*.txt'):
		if filename == 'stats.txt':
			continue
		with open(filename, 'rb') as readfile:
			shutil.copyfileobj(readfile, outfile)

	

