## Main Class
## Author : Danny B
## Date : 02-June-2020
## Func : Main Class

import numpy as np
import csv
from dwt import dwt

x = np.array([])
y = np.array([])
filename = '211.csv'

##Reading file and saving to variables
with open(filename) as csvfile:
    f_read = csv.reader(csvfile, delimiter = ',')
    for row in f_read:
        #taking x parameter only for plotting the signal
        #x = np.append(x, [float(j) for j in row]) -- Comented as only using 1 value
        x = np.append(x, float(row[0]))

        #Cleaning signal data
        if row[2][:1] == '-' : 
            if row[2][1:len(row[2])] == '' :
                y = np.append(y, 0)
            else :
                curr = float(row[2][1:len(row[2])])
                curr = float(-curr)
                y = np.append(y, curr)
        else :
            y = np.append(y, float(row[2]))

feature_dwt = dwt(y)

print(feature_dwt)
