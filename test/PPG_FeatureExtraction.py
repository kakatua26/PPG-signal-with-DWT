# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:45:31 2019

@author: cumitempur
"""
#from scipy.signal import argrelmax, argrelmin
import numpy as np 
from libraries import *
import os
from time import time
"""
Main Program
"""

#ppg_normal, ppg_pac, ppg_pvc = annotation_to_ppg_signal("/home/ipat/Project/TA/KODING_FINAL/data","/home/ipat/Project/TA/KODING_FINAL/annotation","212")
#all_feature = []

lokasi_file = "fitur/gabungan"

for filename in os.listdir("annotation"):
    if (".csv" in filename):
        start = time()
        nomor = filename.replace(".csv","")
        print(nomor)
        try:
            ppg_signal, ppg_class = annotation_to_ppg_signal_labeled("data","annotation",nomor)
            print(ppg_signal)
#            ganti ganti
            ppg_feature = feature_extraction_time_domain_features_and_sliding_window_and_qt(ppg_signal,preprocess=True)

            for i in range(len(ppg_feature)):
                ppg_feature[i].append(ppg_class[i])
            save_feature_to_csv(lokasi_file,nomor,ppg_feature)
            print(time() - start, "second "+lokasi_file)
        except Exception as e:
            print("failed",e)
            print(time() - start, "second "+lokasi_file)
            
