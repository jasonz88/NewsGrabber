def normalizeInput(y):
	n = len(y[0])
	m = len(y)
	for j in range(n):
		mSum = 0
		mMin = 9999
		mMax = -9999
		for i in range(m):
			mSum = mSum + y[i][j]
			if mMin > y[i][j]:
				mMin = y[i][j]
			if mMax < y[i][j]:
				mMax = y[i][j]
		mAvg = mSum / m
		mRange = mMax - mMin
		for i in range(m):
			y[i][j] = (y[i][j] - mAvg) / mRange

