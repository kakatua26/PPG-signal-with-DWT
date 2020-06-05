## Main Class
## Author : Danny B
## Date : 02-June-2020
## Func : Callable DWT Function. Return Feature of the signal

import numpy as np
import statistics
import pywt
from scipy.stats import kurtosis

def comp_moment(feature):
    step = int(len(feature)/2)
    avg_temp = np.zeros([2])
    stn_dev_temp = np.zeros([2])
    kurto_temp = np.zeros([2])
    for i in range(int(len(feature)/step)):
        avg_temp[i] = np.mean(feature[step*i:step*(i+1)])
        stn_dev_temp[i] = statistics.stdev(feature[step*i:step*(i+1)])
        kurto_temp[i] = kurtosis(feature[step*i:step*(i+1)])
        return (avg_temp, stn_dev_temp, kurto_temp)

def dwt(y):
    [a, d1, d2, d3, d4] = pywt.wavedec(y, 'haar', level=4)
    
    #Approximation coeficient
    avg_temp, stn_dev_temp, kurto_temp = comp_moment(a)
    avg = avg_temp
    stn_dev = stn_dev_temp
    kurto = kurto_temp

    # d1 coffiecient
    avg_temp, stn_dev_temp, kurto_temp = comp_moment(d1)
    avg = np.append(avg, avg_temp)
    stn_dev = np.append(stn_dev, stn_dev_temp)
    kurto = np.append(kurto, kurto_temp)

    # d2 coffiecient
    avg_temp, stn_dev_temp, kurto_temp = comp_moment(d2)
    avg = np.append(avg, avg_temp)
    stn_dev = np.append(stn_dev, stn_dev_temp)
    kurto = np.append(kurto, kurto_temp)

    # d3 coffiecient
    avg_temp, stn_dev_temp, kurto_temp = comp_moment(d3)
    avg = np.append(avg, avg_temp)
    stn_dev = np.append(stn_dev, stn_dev_temp)
    kurto = np.append(kurto, kurto_temp)

    # d4 coffiecient
    avg_temp, stn_dev_temp, kurto_temp = comp_moment(d4)
    avg = np.append(avg, avg_temp)
    stn_dev = np.append(stn_dev, stn_dev_temp)
    kurto = np.append(kurto, kurto_temp)

    feature_dwt = np.append(np.append(avg, stn_dev), kurto)

    return feature_dwt
