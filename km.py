from sklearn.cluster import KMeans

km = KMeans(n_clusters=8, init='k-means++', max_iter=100, n_init=1,
                verbose=0)
X=[
 [91.38, 89.67, 92.06, 89.05, 171.01, 1.71],
[89.84, 90.98, 91.40, 89.02, 209.65, -1.40],
[91.12, 91.27, 91.53, 90.80, 117.85, -0.40],
[91.49, 90.79, 91.67, 90.75, 104.42, 0.54],
[91.00, 91.07, 91.88, 90.51, 104.32, -0.56],
[91.51, 90.61, 91.63, 90.33, 145.01, 1.14],
[90.48, 89.99, 90.80, 89.85, 187.39, 0.73],
[89.82, 89.22, 90.10, 89.06, 187.32, 1.14],
[88.81, 88.18, 88.98, 87.26, 33.05, 0.90],
[88.02, 87.87, 88.52, 87.01, 152.40, 0.36],
[87.70, 88.48, 88.65, 87.63, 221.15, -1.04],
[88.62, 88.28, 89.09, 86.83, 355.06, 0.39],
[88.28, 88.19, 88.95, 87.74, 281.85, -0.37],
[88.61, 87.68, 89.49, 87.44, 329.77, 0.93],
[87.79, 88.51, 89.00, 87.10, 313.84, -0.66],
[88.37, 88.55, 89.42, 87.71, 353.96, 0.10],
[88.28, 88.35, 88.99, 87.33, 383.14, -0.46],
[88.69, 88.92, 90.76, 88.04, 462.69, -0.77],
[89.38, 89.44, 89.76, 88.56, 293.84, 0.21],
[89.19, 87.94, 89.49, 87.14, 354.64, 1.35],
[88.00, 86.80, 88.13, 86.27, 330.64, 1.44],
[86.75, 83.66, 86.95, 83.63, 340.74, 3.14],
[84.11, 85.81, 85.90, 83.55, 338.36, -1.89],
[85.73, 83.90, 85.90, 83.59, 335.80, 2.35],
[83.76, 84.21, 84.53, 82.78, 174.18, -0.12],
[83.86, 80.99, 84.25, 80.97, 324.51, 3.21],
[81.25, 81.61, 82.10, 80.28, 351.86, -0.60],
[81.74, 82.15, 82.87, 80.68, 290.22, 0.28],
[81.51, 82.24, 82.75, 80.59, 36.97, -0.42],
[81.85, 80.45, 82.35, 80.44, 160.41, 1.75],
[80.44, 82.44, 82.67, 80.06, 334.53, -2.31],
[82.34, 84.58, 84.74, 82.03, 454.83, -2.97],
[84.86, 84.87, 85.77, 84.48, 315.26, -0.02],
[84.88, 87.73, 87.85, 84.52, 478.82, -3.34],
[87.81, 87.85, 88.63, 87.54, 343.01, 0.00],
[87.81, 86.45, 88.21, 86.10, 472.84, 1.26],
[86.72, 86.75, 87.63, 85.48, 368.00, -0.39],
[87.06, 87.39, 87.49, 85.96, 309.97, 0.24],
[86.85, 86.60, 87.43, 85.96, 315.28, 0.42],
[86.49, 85.09, 86.83, 84.92, 318.00, 2.13],
[84.69, 84.37, 85.36, 83.57, 393.74, 0.94],
[83.90, 82.88, 84.47, 82.83, 281.83, 1.15],
[82.95, 81.45, 83.86, 81.32, 358.54, 1.87],
[81.43, 81.92, 82.12, 80.56, 313.87, -0.91],
[82.18, 81.99, 82.64, 81.50, 296.75, 0.29],
[81.94, 82.48, 82.69, 80.52, 375.25, -0.74],
[82.55, 82.47, 82.88, 81.81, 265.93, 0.04],
[82.52, 82.01, 83.28, 81.45, 339.40, 1.02],
[81.69, 80.73, 82.07, 80.41, 256.38, 1.40],
[80.56, 82.55, 82.70, 80.09, 378.29, -1.48],
[81.77, 79.48, 82.03, 79.35, 40.71, 2.87],
[79.49, 82.95, 83.21, 79.25, 151.36, -4.32],
[83.08, 81.38, 83.28, 80.35, 225.08, 2.25],
[81.25, 82.77, 83.33, 80.75, 345.84, -1.74],
[82.69, 83.23, 84.12, 82.21, 340.96, -0.39],
[83.01, 81.72, 83.45, 81.68, 330.29, 1.64],
[81.67, 81.94, 82.33, 80.88, 333.58, -0.66],
[82.21, 82.95, 83.50, 81.82, 234.32, -0.54],
[82.66, 81.43, 83.13, 80.30, 414.08, 1.21],
[81.67, 83.27, 84.43, 81.00, 450.95, -1.87],
[83.23, 82.60, 84.09, 82.29, 349.17, 0.50],
[82.82, 81.37, 82.99, 81.15, 336.83, 1.66],
[81.47, 81.68, 82.38, 80.77, 313.31, -0.13],
[81.58, 79.84, 81.75, 79.70, 359.94, 2.01],
[79.97, 77.91, 80.18, 77.55, 418.63, 2.71],
[77.86, 76.20, 78.13, 75.60, 388.99, 2.21],
[76.18, 76.23, 77.12, 75.53, 329.35, -0.44],
[76.52, 76.47, 77.17, 75.52, 289.64, 0.04],
[76.49, 74.95, 76.68, 74.66, 288.50, 1.74],
[75.18, 74.84, 75.61, 73.58, 295.90, 0.63],
[74.71, 74.79, 76.00, 73.84, 343.85, 1.62],
[73.52, 74.57, 74.60, 72.81, 34.58, -1.79],
[74.86, 73.59, 75.45, 73.32, 132.43, 1.63],
[73.66, 74.47, 75.25, 72.75, 195.83, -1.22],
[74.57, 75.76, 75.99, 74.11, 289.28, -1.91],
[76.02, 76.64, 76.65, 74.66, 356.33, -1.02],
[76.80, 77.17, 77.99, 76.21, 416.11, -0.51],
[77.19, 76.36, 78.04, 76.36, 349.16, 0.97],
[76.45, 74.40, 76.73, 74.37, 431.37, 2.96],
[74.25, 74.92, 75.96, 73.88, 372.88, -0.56],
[74.67, 73.91, 75.39, 73.37, 359.76, 0.78],
[74.09, 74.30, 74.63, 72.63, 406.64, -0.68],
[74.60, 74.93, 75.44, 73.20, 379.30, -0.56],
[75.02, 73.91, 75.14, 73.11, 340.17, 1.50],
[73.91, 71.70, 74.48, 71.67, 396.61, 2.77],
[71.92, 74.07, 74.73, 71.53, 469.41, -3.72],
[74.70, 75.50, 75.58, 74.01, 260.67, -0.63],
[75.17, 73.18, 75.59, 72.04, 477.90, 2.47],
[73.36, 72.82, 73.98, 72.54, 362.83, 1.16],
[72.52, 71.47, 72.97, 70.76, 343.41, 1.24],
[71.63, 72.95, 73.05, 71.32, 340.99, -2.01],
[73.10, 73.90, 74.48, 72.75, 252.44, -0.49],
[73.46, 74.45, 74.60, 73.19, 33.38, -1.30],
[74.43, 75.27, 76.10, 73.96, 123.22, -1.31],
[75.42, 75.45, 75.74, 73.83, 247.10, -0.46],
[75.77, 75.06, 76.63, 75.01, 347.43, 0.70],
[75.24, 75.60, 75.95, 74.86, 237.46, -0.20],
[75.39, 75.75, 76.74, 75.01, 286.13, -0.46],
[75.74, 77.31, 77.97, 75.52, 430.79, -2.92],
[78.02, 80.22, 80.44, 77.25, 416.89, -2.78],
[80.25, 81.44, 81.62, 79.20, 385.85, -1.51],
[81.48, 80.91, 81.76, 80.71, 267.08, 0.97],
[80.70, 82.12, 82.67, 80.04, 397.53, -1.60],
[82.01, 82.44, 82.48, 81.56, 234.81, -0.56],
[82.47, 82.42, 82.97, 81.62, 293.32, -0.10],
[82.55, 81.41, 82.64, 81.11, 290.98, 1.49],
[81.34, 78.95, 81.77, 78.83, 306.10, 3.03],
[78.95, 78.25, 79.05, 76.83, 294.43, 0.75],
[78.36, 76.89, 78.89, 76.45, 350.58, 1.78],
[76.99, 77.08, 77.74, 75.90, 314.17, -0.66],
[77.50, 78.92, 79.69, 76.79, 377.84, -1.87],
[78.98, 78.98, 79.33, 78.06, 227.66, 0.00],
[78.98, 79.27, 79.60, 78.40, 259.74, -0.40],
[79.30, 76.43, 79.42, 76.16, 358.82, 3.58],
[76.56, 77.85, 78.57, 76.20, 322.41, -1.14],
[77.44, 76.40, 77.57, 75.65, 34.22, 1.18],
[76.54, 75.72, 77.69, 75.50, 169.55, 0.70],
[76.01, 76.82, 77.15, 75.25, 225.25, -0.80],
[76.62, 76.76, 77.66, 75.33, 366.02, -0.55],
[77.04, 77.13, 78.15, 76.38, 364.47, -0.14],
[77.15, 75.06, 77.37, 74.25, 361.79, 2.94],
[74.95, 76.30, 76.43, 74.52, 309.17, -1.50],
[76.09, 75.85, 76.48, 75.00, 269.89, 0.86],
[75.44, 74.85, 76.00, 74.38, 325.81, 1.85],
[74.07, 72.07, 74.92, 71.44, 288.66, 2.90],
[71.98, 72.06, 73.86, 71.09, 325.73, -0.22],
[72.14, 72.67, 73.38, 71.62, 256.17, -1.11],
[72.95, 75.37, 75.40, 72.05, 405.08, -3.54],
[75.63, 75.46, 76.83, 74.39, 357.75, -0.41],
[75.94, 78.15, 78.32, 75.21, 323.27, -2.95],
[78.25, 79.00, 79.38, 77.72, 223.31, -0.77],
[78.86, 76.56, 79.19, 75.90, 316.46, 3.07],
[76.51, 76.00, 76.57, 75.32, 260.40, 0.21],
[76.35, 77.52, 77.83, 75.17, 365.04, -1.11],
[77.21, 77.35, 78.10, 76.53, 27.34, -0.78],
[77.82, 77.50, 78.92, 76.88, 124.29, 0.83],
[77.18, 76.55, 77.45, 75.56, 157.43, 0.51],
[76.79, 77.45, 77.79, 76.17, 303.28, -1.13],
[77.67, 77.06, 78.13, 76.06, 323.36, 0.95],
[76.94, 74.77, 77.16, 74.62, 308.29, 2.42],
[75.12, 74.06, 75.99, 74.04, 321.67, 1.82],
[73.78, 75.52, 75.64, 73.26, 421.74, -2.25],
[75.48, 73.87, 76.30, 73.72, 430.04, 1.48],
[74.38, 72.51, 74.96, 72.03, 436.98, 3.32],
[71.99, 71.16, 72.60, 70.75, 421.97, 0.77],
[71.44, 70.35, 72.49, 69.51, 417.31, -0.10],
[71.51, 74.61, 75.42, 70.73, 448.49, -4.15],
[74.61, 73.70, 74.95, 72.32, 439.43, 2.40],
[72.86, 71.94, 73.93, 71.68, 391.03, 0.39],
[72.58, 73.97, 75.33, 71.64, 438.59, -1.88],
[73.97, 74.90, 75.72, 73.13, 420.67, -0.78],
[74.55, 70.74, 74.95, 70.67, 401.81, 4.25],
[71.51, 70.06, 71.70, 69.21, 405.52, 4.01],
[68.75, 69.90, 70.04, 67.15, 395.40, -2.08],
[70.21, 70.62, 70.96, 69.57, 279.82, 0.24],
[70.04, 69.52, 71.23, 69.00, 449.43, 2.98],
[68.01, 71.20, 71.29, 64.24, 33.22, -2.66],
[69.87, 69.11, 71.43, 67.90, 129.17, 0.66],
[69.41, 70.50, 72.52, 68.91, 206.38, -0.96],
[70.08, 71.79, 72.25, 69.27, 355.83, -2.14],
[71.61, 73.99, 74.13, 70.83, 373.98, -3.75],
[74.40, 75.47, 76.45, 73.62, 532.33, -1.65],
[75.65, 75.89, 77.00, 74.75, 475.85, -0.94],
[76.37, 77.26, 77.68, 75.36, 441.54, -0.56],
[76.80, 76.11, 78.51, 75.80, 447.16, 2.25],
[75.11, 76.95, 78.19, 74.51, 601.53, -2.59],
[77.11, 79.63, 80.39, 74.58, 605.01, -3.58],
[79.97, 82.13, 82.83, 79.15, 512.29, -3.35],
[82.74, 86.09, 86.24, 82.05, 497.43, -4.00],
[86.19, 86.20, 87.15, 85.83, 324.73, 0.05],
[86.15, 85.58, 86.50, 85.16, 392.14, 1.15],
[85.17, 83.33, 85.63, 83.01, 393.69, 2.34],
[83.22, 81.84, 83.44, 81.29, 400.70, 0.95],
[82.44, 83.89, 84.33, 81.70, 462.84, -2.09],
[84.20, 85.22, 85.63, 83.73, 337.12, -1.08],
[85.12, 83.75, 85.19, 82.92, 322.55, 1.70],
[83.70, 83.47, 84.07, 81.73, 433.95, 0.02],
[83.68, 84.03, 84.64, 82.92, 428.24, 0.28],
[83.45, 81.69, 83.65, 81.51, 33.43, 2.46],
[81.45, 82.92, 83.00, 80.53, 122.83, -2.15],
[83.24, 85.35, 85.44, 82.52, 268.76, -2.65],
[85.51, 85.91, 86.27, 85.27, 274.10, -0.38],
[85.84, 83.82, 86.39, 83.71, 425.17, 2.13],
[84.05, 84.36, 84.42, 82.51, 552.30, -0.34],
[84.34, 85.17, 85.71, 84.08, 421.90, -0.68],
[84.92, 85.56, 86.37, 84.12, 442.94, -0.55],
[85.39, 85.64, 85.88, 84.38, 389.60, -0.57],
[85.88, 86.76, 87.00, 85.52, 379.53, -1.11],
[86.84, 86.74, 87.09, 86.13, 290.48, 0.25],
[86.62, 85.31, 86.90, 85.06, 244.11, 2.06],
[84.87, 83.36, 85.37, 83.21, 234.24, 1.33],
[83.76, 82.50, 83.85, 82.22, 371.96, 1.69],
[82.37, 82.50, 82.74, 81.77, 244.44, 0.24],
[82.17, 80.24, 82.78, 80.18, 311.04, 2.71],
[80.00, 80.13, 81.46, 79.54, 324.61, -0.66],
[80.53, 80.25, 81.48, 80.10, 296.27, -0.10],
[80.61, 81.58, 81.64, 79.88, 307.76, -1.59],
[81.91, 81.81, 82.20, 80.85, 261.87, 0.81],
[81.25, 80.93, 81.49, 78.57, 32.68, 0.71],
[80.68, 82.17, 82.17, 79.86, 150.68, -1.85],
[82.20, 82.85, 82.85, 81.68, 208.34, -0.88],
[82.93, 82.05, 83.09, 81.72, 275.21, 1.51],
[81.70, 79.84, 82.10, 79.32, 290.18, 2.38],
[79.80, 81.13, 81.31, 79.13, 283.14, -1.77],
[81.24, 82.20, 83.16, 80.57, 337.20, -1.06],
[82.11, 81.98, 82.38, 81.33, 279.16, 0.02],
[82.09, 81.38, 83.03, 80.81, 430.15, 0.74],
[81.49, 81.75, 81.91, 80.16, 332.13, -0.46],
[81.87, 81.79, 82.41, 80.75, 295.35, 0.45],
[81.50, 80.58, 82.07, 80.47, 299.63, 1.61],
[80.21, 81.01, 81.09, 79.70, 278.89, -0.82],
[80.87, 79.64, 81.23, 79.44, 306.91, 1.49],
[79.68, 78.88, 80.95, 78.26, 301.70, 1.25],
[78.70, 79.84, 80.62, 78.06, 298.72, -1.21],
[79.66, 78.33, 80.05, 77.82, 319.04, 1.91],
[78.17, 80.31, 80.32, 77.05, 346.03, -2.29],
[80.00, 79.13, 80.45, 78.25, 289.32, 1.45],
[78.86, 80.03, 80.39, 78.22, 316.88, -1.62],
[80.16, 80.10, 80.51, 79.45, 34.94, 0.44],
[79.81, 78.37, 80.10, 77.76, 183.23, 0.95],
[79.06, 77.29, 79.29, 76.32, 274.17, 2.24],
[77.33, 77.32, 77.82, 76.53, 283.82, 0.42],
[77.01, 74.02, 77.42, 73.71, 365.03, 3.89],
[74.13, 75.30, 75.35, 72.66, 335.81, -1.53],
[75.28, 74.65, 75.69, 73.38, 464.44, 1.02],
[74.52, 73.79, 74.97, 72.60, 422.06, 1.04],
[73.75, 71.68, 74.15, 71.32, 429.85, 2.59],
[71.89, 72.18, 72.39, 70.77, 345.64, 0.98],
[71.19, 73.04, 73.94, 69.50, 596.14, -2.67],
[73.14, 77.09, 77.17, 72.42, 523.89, -4.99],
[76.98, 76.99, 78.04, 76.52, 391.29, -0.32],
[77.23, 74.94, 77.41, 74.40, 366.27, 3.76],
[74.43, 72.84, 74.99, 72.49, 277.41, 2.11],
[72.89, 73.89, 74.82, 72.43, 335.27, -1.02],
[73.64, 73.75, 74.49, 72.93, 292.91, -0.04],
[73.67, 74.69, 75.09, 72.65, 356.18, -1.39],
[74.71, 75.28, 75.39, 73.82, 256.21, -0.73],
[75.26, 74.24, 75.42, 74.06, 280.56, 0.97],
[74.54, 75.84, 76.50, 74.01, 348.82, -2.02],
[76.08, 77.56, 78.36, 75.66, 327.42, -1.98],
[77.62, 78.88, 79.03, 76.96, 40.68, -1.77],
[79.02, 77.85, 79.15, 76.76, 186.88, 1.31],
[78.00, 79.20, 79.31, 77.70, 200.56, -1.75],
[79.39, 79.63, 80.36, 78.92, 275.40, -0.33],
[79.65, 80.06, 80.67, 78.37, 401.63, -1.41],
[80.79, 82.07, 82.34, 79.91, 333.87, -2.10],
[82.52, 82.88, 83.95, 81.96, 296.30, -0.28],
[82.75, 82.65, 83.47, 81.80, 310.38, 0.11],
[82.66, 83.20, 83.36, 82.26, 246.63, -0.63],
[83.18, 81.43, 83.52, 80.85, 370.06, 1.72],
[81.77, 81.63, 82.00, 80.95, 258.89, 0.32],
[81.51, 79.63, 81.79, 79.63, 263.54, 2.71]]


print X[::-1]
km.fit(X[::-1])
y_train = km.labels_[1:]
cc = km.cluster_centers_


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






from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
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
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


###############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(input='filename',stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(trainingfiles)
else:
    vectorizer = TfidfVectorizer(input='filename',sublinear_tf=True, max_df=0.5,min_df=15
                                 stop_words='english')
    X_train = vectorizer.fit_transform(trainingfiles)
duration = time() - t0


data_train_size_mb = size_mb(X_train)
data_test_size_mb = size_mb(X_test)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target


data_train_size_mb = size_mb(X_train)
data_test_size_mb = size_mb(data_test.data)
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
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
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(vectorizer.get_feature_names())


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
