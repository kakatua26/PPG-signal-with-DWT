
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
Import classifier
"""
skenario = "fitur_skenario_3/"
nama_fitur = "gabungan"

fitur_path = skenario+nama_fitur
lokasi_model = "model_skenario/"+skenario

import pickle
clf = pickle.load(open(lokasi_model+"/KNN_7_"+nama_fitur+".sav","rb"))
"""
Generate Features and Classes
"""
hasil = []
for filename in os.listdir(fitur_path):
    if (".csv" in filename):
        start = time()
        nomor = filename.replace(".csv","")
        print(nomor)
        try:
            feature_list, class_list = read_feature_from_csv(fitur_path+"/%s.csv" % nomor)
            pred = clf.predict(feature_list)
            hasil.append(generate_confusion_matrix(nomor,class_list,pred,smoothing=False))
        except Exception as e:
            print("failed",e)
            print(time() - start, "second")

"""
Export Result to CSV
"""
columns_list = ["Sample_Data", "Normal","PAC","PVC","pred_Normal", "pred_PAC","pred_PVC","True_Positive_PAC","False_Positive_PAC","True_Negative_PAC","False_Negative_PAC","Accuracy_PAC","Spesifisity_PAC","sn_PAC","True_Positive_PVC", "False_Positive_PVC", "True_Negative_PVC", "False_Negative_PVC","Accuracy_PVC","Spesifisity_PVC","sn_PVC"]
pd.DataFrame(hasil,columns=columns_list).to_csv("hasil_skenario/"+fitur_path+"/confusion_matrix_fitur_"+nama_fitur+".csv",index=False)

data = pd.DataFrame(hasil)
