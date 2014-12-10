import numpy as np  # a conventional alias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
import os,sys,glob

filenames = []
rootdir = sys.argv[1]
for subdir, dirs, files in os.walk(rootdir):
	if '-' not in subdir:
		continue
	if os.path.isfile(subdir+'/out.data'):
		filenames.append(subdir+'/out.data')
	
vectorizer = TfidfVectorizer(input='filename',stop_words='english',min_df=15)
dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
vocab = vectorizer.get_feature_names()  # a list

print vectorizer.get_stop_words()