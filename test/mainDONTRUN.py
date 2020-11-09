## Main Class
## Author : Danny B
## Date : 02-June-2020
## Func : Main Class

import numpy as np
import csv, os, glob, ntpath
from dwt import dwt
import pandas as pd
from libraries import *

x = np.array([])
y = np.array([])
path = 'D:\\TA ORANG\\Ryan\\test\\data'
pathname = "D:\\TA ORANG\\Ryan\\test\\data\\"
#os.chdir(path)
#files = glob.glob('*.{}'.format('csv'))
files = glob.glob(os.path.join(path, '*.csv'))
file_path = "fitur/dwt"

##Reading file and saving to variables
for filename in files:
    with open(filename) as csvfile:
        print(filename)
        f_read = csv.reader(csvfile, delimiter = ',')
        number = filename.replace(".csv","")
        number = number.replace(pathname,"")
        print(number)
        #ppg_signal, ppg_class = annotation_to_ppg_signal_labeled("data","annotation",number)
        #print(ppg_signal)
        for row in f_read:
            if (row[0] != "'sample interval'" and row[0] != "'0.002 sec'") : 
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

        #save_feature_to_csv(file_path,number,feature_dwt)
        dwt_feature = pd.DataFrame(feature_dwt)
        dwt_feature.to_csv("%s/%s.csv" % (file_path, number),index=False,header=False)
        #print(feature_dwt)
