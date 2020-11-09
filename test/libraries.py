#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:45:01 2019

@author: ipat
"""
import pywt
import numpy as np 
import pandas as pd
from scipy.signal import find_peaks, resample
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


#variable global
sample_rate = 500
scaler = MinMaxScaler()

"""
Fungsi Lain
"""
def plot_color_text(filtered,predicted_beats,start_stop):
    minimum = min(filtered)
    maximum = max(filtered)
    lel = [np.arange(data[0],data[1]) for data in start_stop]
    #plt.plot(ecg_raw)
    plt.plot(filtered)
    for i in range(len(predicted_beats)):
        if (predicted_beats[i] != "N"):
            plt.fill_between(lel[i],minimum,maximum,facecolor='red', alpha=0.5)
            plt.text(start_stop[i][0],maximum,predicted_beats[i])
#    plt.scatter(peaks, [filtered[peaks[i]] for i in range(len(peaks))],c='red')
    plt.show()
    
def plot_with_rpeaks(filtered,r_peaks):
    peaks = [filtered[peak] for peak in r_peaks]
    plt.plot(filtered)
    plt.scatter(r_peaks,peaks,c='red')
#    plt.savefig("sinyal_clean_peaks",dpi=1000)
    plt.show()
"""
END FUNGSI LAIN
"""

"""
PREPROCESSING
"""
def find_signal_peaks(arraynya, minimum=0, maximum=None, freq=500):
    dist = freq/2
    r_peaks = find_peaks(arraynya,distance=dist,prominence=(minimum,maximum))
    return r_peaks[0].tolist()

def get_rr(r_peaks, to_sec=False,sample_rate=125):
    rr_list = []
    start_stop = []
    for i in range(len(r_peaks)-2):
        rr_list.append(r_peaks[i+1]-r_peaks[i])
        start_stop.append([r_peaks[i],r_peaks[i+1]])
    if (to_sec):
        rr_list = np.divide(rr_list,sample_rate)
    return rr_list, start_stop

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def baseline_remove_ppg(signal,frequency=500):
    half_freq = int(frequency/2)
    baseline = resample(moving_average(signal,n=half_freq),len(signal))
    return np.subtract(signal,baseline)

def remove_signal(signal):
    return np.zeros_like(signal)

def remove_noise(noisy_signal):
    denoised = noisy_signal
    for i in range(5):
        denoised = pywt.dwt(denoised,'sym8')[0]
    for i in range(5):
        denoised = pywt.idwt(denoised,remove_signal(denoised),wavelet='sym8')
    return denoised

def annotation_to_ppg_signal_split(signal_path, annotation_path, number):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, low_memory=False).replace("-",100)
    annotation = pd.read_csv(anot_file, low_memory=False)
    ppg = signal.iloc[1:,2].values
    ppg_normal, ppg_pac, ppg_pvc = [], [], []
    start, stop, signal_class = annotation["start_index"].values, annotation["stop_index"].values, annotation["signal_class"].values.tolist()
    for i in range(len(annotation)):
        if (signal_class[i] == "N"):
            ppg_normal.append(ppg[start[i]:stop[i]].astype("float64"))
        elif (signal_class[i] == "A"):
            ppg_pac.append(ppg[start[i]:stop[i]].astype("float64"))
        elif (signal_class[i] == "V"):
            ppg_pvc.append(ppg[start[i]:stop[i]].astype("float64"))
    return ppg_normal, ppg_pac, ppg_pvc

def annotation_to_ppg_signal_labeled(signal_path, annotation_path, number):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, low_memory=False).replace("-",100)
    annotation = pd.read_csv(anot_file, low_memory=False)
    ppg = signal.iloc[1:,2].values
    ppg_signal = []
    start, stop, signal_class = annotation["start_index"].values, annotation["stop_index"].values, annotation["signal_class"].values.tolist()
    for i in range(len(annotation)):
            ppg_cut = ppg[start[i]:stop[i]].astype("float64").tolist()
            ppg_signal.append(ppg_cut)
    return ppg_signal, signal_class


def read_feature_from_csv(file_name):
    feature = pd.read_csv(file_name,low_memory=False).values
    pp_list = feature[:,0:-1]
    class_list = feature[:,-1]
    return pp_list, class_list

def read_feature_from_csv1(file_name):
    feature = pd.read_csv(file_name,low_memory=False)
    pp_list = feature.iloc[:,0:-1]
    class_list = feature.iloc[:,-1]
    return pp_list, class_list
    
def preprocess_ppg_signal(input_signal,sample_rate=500):
    baseline_removed = baseline_remove_ppg(input_signal,frequency=sample_rate)
    clean_ppg = remove_noise(baseline_removed)
    return clean_ppg

    
    
    

"""
END PREPROCESSING
"""    

"""
FITUR EKSTRAKSI
"""

#def feature_extraction_ez(signal_array):
#    features = []
#    for signal in signal_array:
#        min_signal, max_signal, mean_signal, std_signal = np.min(signal), np.max(signal), np.mean(signal), np.std(signal)
#        feature = [min_signal, max_signal, mean_signal, std_signal]
#        features.append(feature)
#    return features
#
#def feature_extraction_medium(signal_array,window_count=4,sample_rate=500):
#    features = []
#    for signal in signal_array:
#        min_signal, max_signal, mean_signal, std_signal = np.min(signal), np.max(signal), np.mean(signal), np.std(signal)
#        peaks = find_signal_peaks(signal,minimum=max_signal*0.4)
#        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
#        middle = int(len(rr_list)/2)
#        
#        if (len(rr_list[middle:middle+window_count]) < window_count):
#            rr_list = rr_list[middle-1:middle+window_count-1]
#        else:
#            rr_list = rr_list[middle:middle+window_count]
#        
#        feature = [min_signal, max_signal, mean_signal, std_signal]
#        feature = np.concatenate([feature,rr_list])
#        features.append(feature)
#    return features
#

#QT INTERVAL FITUR
def secondDerivative(signal):
    new = np.zeros_like(signal)
    for i in range(1,len(signal)-1):
        new[i] = (signal[i+1] - (2 * signal[i]) + signal[i-1]) / (len(signal)**2)
    return new   

def detect_qt(data_peak ,to_sec=True, sample_rate=500):
    qt_interval = []
    start_stop = []
    for i in range(1,len(data_peak)-1):
        peak_sebelum = data_peak[i-1]
        peak = data_peak[i]
        peak_sesudah = data_peak[i+1]
        
        
        deteksi_15persen  = (peak-peak_sebelum)*0.15
#        deteksi_15persen  = peak*0.15
        peak_q = round(peak - deteksi_15persen)
        
        deteksi_40persen = (peak_sesudah-peak)*0.40
#        deteksi_40persen = peak*0.40

        peak_t = round(peak + deteksi_40persen)
        qt_interval.append(peak_t-peak_q)
        start_stop.append([peak_q,peak_t])
    if(to_sec):
        qt_interval = np.divide(qt_interval,sample_rate)
    return qt_interval , start_stop



def feature_extraction_qt(signal_array,window_count=6,sample_rate=500, preprocess=True):
    features = []
    for signal in signal_array:
        if (preprocess):
            second = secondDerivative(signal)
            signal = preprocess_ppg_signal(second)
            signal *= 10**10
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        qt_list, qt_startstop = detect_qt(peaks,to_sec=True,sample_rate=sample_rate)

        middle = int(len(qt_list)/2)
        kiri = middle-(window_count//2)
        qt_list = qt_list[kiri:kiri+window_count].tolist()
#        print(len(rr_list))
        if (len(qt_list) == window_count):
            features.append(qt_list)
    return features

import math
#time domain features
def feature_extraction_time_domain_features(signal_array,sample_rate=500 , preprocess=False):
    features = []
    i = 0
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
        rata_ppi = np.mean(rr_list)
        std_ppi = np.std(rr_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          print(i)
          rata_ppi = np.nansum(np.mean(rr_list))
          std_ppi = np.nansum(np.std(rr_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal])
        else:
            features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal])
        i = i+1
    return features

#QT interval dengan sinyal second derivative
    
    
#SLIDING WINDOW
def feature_extraction_pp(signal_array,window_count=6,sample_rate=500, preprocess=False):
    features = []
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
        middle = int(len(rr_list)/2)
        kiri = middle-(window_count//2)
        rr_list = rr_list[kiri:kiri+window_count].tolist()
#        print(len(rr_list))
        if (len(rr_list) == window_count):
            features.append(rr_list)
    return features
#END SLIDING WINDOW
    
#SKENARIO 2
#TIME DOMAIN + SLIDING WINDOW
def feature_extraction_time_domain_features_and_sliding_window(signal_array,window_count=6,sample_rate=500 , preprocess=False):
    features = []
    for signal in signal_array:
        if (preprocess):
            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        
#        TIME DOMAIN
        rata_ppi = np.mean(rr_list)
        std_ppi = np.std(rr_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        
#        SLIDING WINDOW
        middle = int(len(rr_list)/2)
        kiri = middle-(window_count//2)
        sliding_window = rr_list[kiri:kiri+window_count].astype("float64")
        
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          
          rata_ppi = np.nansum(np.mean(rr_list))
          std_ppi = np.nansum(np.std(rr_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          
          if (len(sliding_window) == window_count):
              array_1 = [rata_ppi,std_ppi,rata_sinyal,std_sinyal]
              array_1.extend(sliding_window)
              features.append(array_1)

#          features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal]+sliding_window)
        else:
            if (len(sliding_window) == window_count):
                  array_1 = [rata_ppi,std_ppi,rata_sinyal,std_sinyal]
                  array_1.extend(sliding_window)
                  features.append(array_1)    

#                features.append([rata_ppi,std_ppi,rata_sinyal,std_sinyal]+sliding_window)

    return features
#END TIME DOMAIN + SLIDING WINDOW
    
#TIME DOMAIN + QT
def feature_extraction_time_domain_features_and_qt(signal_array,window_count=6,sample_rate=500 , preprocess=True):
    features = []
    for signal in signal_array:
        if (preprocess):
            second = secondDerivative(signal)
            signal = preprocess_ppg_signal(second,sample_rate=sample_rate)
            signal *= 10**10
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        qt_list, qt_startstop = detect_qt(peaks,to_sec=True,sample_rate=sample_rate)
        
#        TIME DOMAIN
        rata_qt = np.mean(qt_list)
        std_qt = np.std(qt_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        
#        QT
        middle = int(len(qt_list)/2)
        kiri = middle-(window_count//2)
        qt_list = qt_list[kiri:kiri+window_count].astype("float64")
        
        if(math.isnan(rata_qt) or math.isnan(std_qt) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          
          rata_qt = np.nansum(np.mean(qt_list))
          std_qt = np.nansum(np.std(qt_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          if (len(qt_list) == window_count):
              array_1 = [rata_qt,std_qt,rata_sinyal,std_sinyal]
              array_1.extend(qt_list)
              features.append(array_1)
          
#          features.append([rata_qt,std_qt,rata_sinyal,std_sinyal]+qt_list)
          
        else:
            if (len(qt_list) == window_count):
                array_1 = [rata_qt,std_qt,rata_sinyal,std_sinyal]
                array_1.extend(qt_list)
                features.append(array_1)
    return features
#END TIME DOMAIN + QT
#END SKENARIO 2
    

#SKENARIO 3
#TIME DOMAIN + SLIDING WINDOW + QT INTERVAL
def feature_extraction_time_domain_features_and_sliding_window_and_qt(signal_array,window_count=6,sample_rate=500 , preprocess=True):
    features = []
    for signal in signal_array:
        if (preprocess):
            second = secondDerivative(signal)
            signal = preprocess_ppg_signal(second,sample_rate=sample_rate)
            signal *= 10**10
            
        peaks = find_signal_peaks(signal,minimum=0.2)
        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
        qt_list, qt_startstop = detect_qt(peaks,to_sec=True,sample_rate=sample_rate)

        
#        SLIDING WINDOW
        middle = int(len(rr_list)/2)
        kiri = middle-(window_count//2)
        sliding_window = rr_list[kiri:kiri+window_count].astype("float64")
        
#QT            
        middle = int(len(qt_list)/2)
        kiri = middle-(window_count//2)
        qt_list = qt_list[kiri:kiri+window_count].astype("float64")

#        TIME DOMAIN
        rata_ppi = np.mean(sliding_window)
        std_ppi = np.std(sliding_window)
        rata_qt = np.mean(qt_list)
        std_qt = np.std(qt_list)
        rata_sinyal = np.mean(signal)
        std_sinyal = np.std(signal)
        
        if(math.isnan(rata_ppi) or math.isnan(std_ppi) or math.isnan(rata_qt) or math.isnan(std_qt) or math.isnan(rata_sinyal) or math.isnan(std_sinyal)):
          print("TRUE")
          rata_ppi = np.nansum(np.mean(sliding_window))
          std_ppi = np.nansum(np.std(sliding_window))
          rata_qt = np.nansum(np.mean(qt_list))
          std_qt = np.nansum(np.std(qt_list))
          rata_sinyal = np.nansum(np.mean(signal))
          std_sinyal = np.nansum(np.std(signal))
          if (len(sliding_window) == window_count and len(qt_list) == window_count):
              array_1 = [rata_ppi,std_ppi,rata_qt,std_qt,rata_sinyal,std_sinyal]
              array_1.extend(sliding_window)
              array_1.extend(qt_list)
              features.append(array_1)
#          features.append([rata_ppi,std_ppi,rata_qt,std_qt,rata_sinyal,std_sinyal]+sliding_window+qt_list)
          
        else:
            if (len(sliding_window) == window_count and len(qt_list) == window_count):
                array_1 = [rata_ppi,std_ppi,rata_qt,std_qt,rata_sinyal,std_sinyal]
                array_1.extend(sliding_window)
                array_1.extend(qt_list)
                features.append(array_1)
                
#                features.append([rata_ppi,std_ppi,rata_qt,std_qt,rata_sinyal,std_sinyal]+sliding_window+qt_list)
    return features
#END TIME DOMAIN + SLIDING WINDOW + QT INTERVAL


def save_feature_to_csv(file_path,number, ppg_feature):
    df_feature = pd.DataFrame(ppg_feature)
#    df_class = pd.DataFrame(ppg_class)
    df_feature.to_csv("%s/%s.csv" % (file_path, number),index=False,header=False)
#    df_class.to_csv("%s/%sclass.csv" % (file_path, number) ,index=False,header=False)
#def feature_extraction_pp(signal_array,window_count=4,sample_rate=500, preprocess=False):
#    features = []
#    for signal in signal_array:
#        if (preprocess):
#            signal = preprocess_ppg_signal(signal,sample_rate=sample_rate)
#            
#        peaks = find_signal_peaks(signal,minimum=0.2)
#        rr_list, rr_startstop = get_rr(peaks,to_sec=True,sample_rate=sample_rate)
##        middle = int(len(rr_list)/2)
#        
##        if (len(rr_list[middle:middle+window_count]) < window_count):
##            rr_list = rr_list[middle-1:middle+window_count-1]
##        else:
##            rr_list = rr_list[middle:middle+window_count]
##        
##        print(rr_list)
##        feature = np.concatenate([feature,rr_list])
#        features.append(rr_list)
#    return features
#    
"""
END FITUR EKSTRAKSI
"""

def class_count(r_class):
#    count = BeatCount()
    kelas = [0,0,0]
    for cl in r_class:
        if (cl == "N"):
            kelas[0] += 1
        elif (cl == "A"):
            kelas[1] += 1
        elif (cl == "V"):
            kelas[2] += 1
    return kelas


"""
Metrics
"""
def generate_original_and_predicted_class(original_class,predicted_class):
    temp = [0,0,0,0,0,0]
    for i in range(len(original_class)):
        if (original_class[i] == "N"):
            temp[0] += 1
        elif (original_class[i] == "A"):
            temp[1] += 1
        elif (original_class[i] == "V"):
            temp[2] += 1
        if (predicted_class[i] == "N"):
            temp[3] += 1
        elif (predicted_class[i] == "A"):
            temp[4] += 1
        elif (predicted_class[i] == "V"):
            temp[5] += 1
    return temp

def generate_confusion_matrix(number,original_class,predicted_class,smoothing=False,smoothing_value=10^-4):
    count_original = class_count(original_class)
    count_predicted = class_count(predicted_class)
    tp_PAC, tn_PAC, fp_PAC, fn_PAC = 0,0,0,0
    tp_PVC,tn_PVC,fp_PVC,fn_PVC = 0,0,0,0
    acc_PAC, acc_PVC = 0,0
    sp_PAC, sp_PVC = 0,0
    sn_PAC, sn_PVC = 0,0
    f1_PAC, f1_PVC = 0,0
    minimum_number = smoothing_value
    for i in range(len(original_class)):
        ori = original_class[i]
        tes = predicted_class[i]
        if (ori == "N"):
            if (tes == "V"):
                fp_PVC += 1
                tn_PAC += 1
            elif (tes == "A"):
                fp_PAC += 1
                tn_PVC += 1
            else:
                tn_PVC += 1
                tn_PAC += 1
        elif (ori == "V"):
            if (tes == "V"):
                tp_PVC += 1
                tn_PAC += 1
            elif (tes == "A"):
                fp_PAC += 1
                fn_PVC += 1
            else:
                tn_PAC += 1
                fn_PVC += 1
        elif (ori == "A"):
            if (tes == "V"):
                fp_PVC += 1
                fn_PAC += 1
            elif (tes == "A"):
                tp_PAC += 1
                tn_PVC += 1
            else:
                tn_PVC += 1
                fn_PAC += 1
    if (smoothing):
        acc_PAC = (tn_PAC+tp_PAC+minimum_number) / (tn_PAC+tp_PAC+fn_PAC+fp_PAC+minimum_number) * 100
        acc_PVC = (tn_PVC+tp_PVC+minimum_number) / (tn_PVC+tp_PVC+fn_PVC+fp_PVC+minimum_number) * 100
        sp_PAC = (tn_PAC+minimum_number)/(tn_PAC+fp_PAC+minimum_number) * 100
        sp_PVC = (tn_PVC+minimum_number)/(tn_PVC+fp_PVC+minimum_number) * 100
        sn_PAC = (tp_PAC+minimum_number) / (tp_PAC+fn_PAC+minimum_number) * 100
        sn_PVC = (tp_PVC+minimum_number) / (tp_PVC+fn_PVC+minimum_number) * 100
    else:
        try:
            acc_PAC = (tn_PAC+tp_PAC) / (tn_PAC+tp_PAC+fn_PAC+fp_PAC) * 100
        except:
            acc_PAC = 0
        try:
            acc_PVC = (tn_PVC+tp_PVC) / (tn_PVC+tp_PVC+fn_PVC+fp_PVC) * 100
        except:
            acc_PVC = 0
        try: 
            sp_PAC = tn_PAC/(tn_PAC+fp_PAC) * 100
        except:
            sp_PAC = 0
        try:
            sp_PVC = tn_PVC/(tn_PVC+fp_PVC) * 100
        except:
            sp_PVC = 0
        try:
            sn_PAC = (tp_PAC) / (tp_PAC+fn_PAC) * 100
        except:
            sn_PAC = 0
        try:
            sn_PVC = (tp_PVC) / (tp_PVC+fn_PVC) * 100
        except:
            sn_PVC = 0
    temp = [number,count_original[0], count_original[1], count_original[2], count_predicted[0], count_predicted[1], count_predicted[2], tp_PAC, fp_PAC, tn_PAC, fn_PAC,round(acc_PAC,2), round(sp_PAC,2),sn_PAC, tp_PVC, fp_PVC, tn_PVC, fn_PVC,round(acc_PVC,2),round(sp_PVC,2),sn_PVC]
    columns = ["Number", "Normal","PAC","PVC","pred_Normal", "pred_PAC","pred_PVC","tp_PAC","fp_PAC","tn_PAC","fn_PAC","acc_PAC","sp_PAC","sn_PAC","tp_PVC", "fp_PVC", "tn_PVC", "fn_PVC","acc_PVC","sp_PVC","sn_PVC"]
#    print(len(temp), len(columns))
    return temp
