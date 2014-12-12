import numpy as np  # a conventional alias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
import os,sys,glob


filenames = []
#filenames = tuple(open(sys.argv[1],'r'))
ins = open( sys.argv[1], "r" )

for line in ins:
    filenames.append( line[:-1] )
ins.close()

#filenames = tuple(open(sys.argv[1],'r'))
ins = open( sys.argv[2], "r" )

for line in ins:
    filenames.append( line[:-1] )
ins.close()

#filenames = tuple(open(sys.argv[1],'r'))
ins = open( sys.argv[3], "r" )

for line in ins:
    filenames.append( line[:-1] )
ins.close()

#filenames = tuple(open(sys.argv[1],'r'))
ins = open( sys.argv[4], "r" )

for line in ins:
    filenames.append( line[:-1] )
ins.close()

import pickle
print len(filenames)
vectorizer = CountVectorizer(input='filename',stop_words='english',min_df=15)
X = vectorizer.fit_transform(filenames)  # a sparse matrix
vocab = vectorizer.get_feature_names()  # a list
# print vectorizer.get_stop_words()
print dtm[333]
fdtm = open('cvt.txt','w')
pickle.dump(dtm,fdtm)
fdtm.close()