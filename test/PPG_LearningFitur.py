
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
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
"""
Generate Features and Classes
"""
all_feature, all_class = [], []

skenario = "fitur/"
nama_fitur = "dwt"

fitur_path = skenario+nama_fitur
model_path = "model_skenario/"+skenario
model_filename = "KNN_7_"+nama_fitur

for filename in os.listdir(fitur_path): 
    if (".csv" in filename):
        start = time()
        nomor = filename.replace(".csv","")
        print(nomor)
        try:
            feature_list, class_list = read_feature_from_csv(fitur_path+"/%s.csv" % nomor)
            if (len(all_feature) == 0):
                all_feature = feature_list
                all_class= class_list
            else:
                all_feature = np.concatenate([all_feature, feature_list])
                all_class = np.concatenate([all_class, class_list])
#            print (class_count(class_list))
            print(time() - start, "second")
        except Exception as e:
            print("failed",e)
            print(time() - start, "second")


"""
Split Train and Test
"""            
from imblearn.over_sampling import SMOTE
ros = SMOTE()
X_resampled, y_resampled = ros.fit_resample(all_feature,all_class)
count_original = class_count(all_class)
count_resampled = class_count(y_resampled)

"""
kFold Algorithm
"""
from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=10) # Define the split - into 2 folds 
accuracy_list = []
for train_index, test_index in kf.split(X_resampled):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]
    
    """
    Import and fit classifier
    """
    #import skflow
    #
#    aktivasi = "tanh"
#    sizes = 12
#    clf = MLPClassifier(activation=aktivasi,verbose=1,hidden_layer_sizes=12,max_iter=300)
    #clf = LogisticRegression()
    clf = KNeighborsClassifier(n_neighbors=7)
    
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    """
    Training Metrics
    """
    accuracy = accuracy_score(y_test,pred)
    accuracy_list.append(accuracy)


average_accuracy = np.average(accuracy_list)
accuracy_list.append(average_accuracy)

akurasi_tocsv=pd.DataFrame(accuracy_list)
akurasi_tocsv.to_csv("%s_akurasi.csv" % (model_path+model_filename),index=False,header=False)
print(average_accuracy)
"""
Save Model To File using Pickle
"""
import pickle
#pickle.dump(clf,open(model_path+"/logistic_regression.sav","wb"))
#pickle.dump(clf,open(model_path+"/ANN_%s_%s.sav" % (sizes, aktivasi),"wb"))
#pickle.dump(clf,open(model_path+"/decision_tree.sav","wb"))
#pickle.dump(clf,open(model_path+"/decision_tree.sav","wb"))

pickle.dump(clf,open(model_path+model_filename+".sav","wb"))
#hehe = generate_original_and_predicted_class(y_test,pred)

