import shutil
import os,sys,glob

rootdir = sys.argv[1]
for subdir, dirs, files in os.walk(rootdir):
	print subdir
	# if '-' not in subdir:
	# 	continue
	# os.chdir(subdir)
	# outfile = open(outfilename, 'wb')
	# for filename in glob.glob('*.txt'):
	# 	if filename == 'stats.txt':
	# 		continue
	# 	with open(filename, 'rb') as readfile:
	# 		shutil.copyfileobj(readfile, outfile)

	

