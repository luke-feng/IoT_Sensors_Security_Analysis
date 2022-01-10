import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn import metrics
import time
import ast
import os,sys
import pickle
import multiprocessing as mp
import psutil
from numpy.core.fromnumeric import mean

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def train(train_data, test_data, clsname):
    X = train_data
    y = [1 for i in range(0,len(X))]
    outliers_fraction = 0.05
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, shuffle=False)
    y_val = [1 for i in range(0,len(X_val))]
    
    classifiers = {
            "Robust covariance": EllipticEnvelope(contamination=outliers_fraction , support_fraction=0.5),
            "One-Class SVM": OneClassSVM(cache_size=200, gamma='scale', kernel='rbf',nu=0.05,  shrinking=True, tol=0.001,verbose=False),
            "SGD One-Class SVM": SGDOneClassSVM(nu=outliers_fraction, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4),
            "Isolation Forest": IsolationForest(contamination=outliers_fraction,random_state=42),
        }

    name = clsname
    clf = classifiers[name]
    t1 =time.time()
    try:
        clf.fit(X_train)
        t2 =time.time()        
        y_pred = clf.predict(X_val)
        val_score = metrics.accuracy_score(y_val,y_pred)
    except:
        t2 =time.time()
        y_pred = []
        val_score = 0
    train_t = t2 -t1

    X_test = test_data
    y_test = [-1  for i in range(0,len(X_test))]

    t1 =time.time()
    clf.predict([X_test[0]])
    t2 =time.time()
    t_test = t2 -t1

    y_pred = clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred)
    return clf, val_score, test_score, train_t, t_test


def run():
    features = [ 
                'system calls frequency_1gram',
                'system calls tfidf_1gram', 
                'system calls hashing_1gram',
                'system calls frequency_2gram', 
                'system calls tfidf_2gram',
                'system calls frequency_3gram',
                'system calls tfidf_3gram',
                'system calls frequency_2gram-pcas', 
                'system calls tfidf_2gram-pcas',
                'system calls frequency_3gram-pcas',
                'system calls tfidf_3gram-pcas', 
                'system calls frequency_4gram-pcas', 
                'system calls tfidf_4gram-pcas',
                'system calls frequency_5gram-pcas',
                'system calls tfidf_5gram-pcas',
                'system calls frequency_1gram-scaled', 
                'system calls tfidf_1gram-scaled',                
                'system calls frequency_2gram-scaled', 
                'system calls tfidf_2gram-scaled',
                'system calls frequency_3gram-scaled',
                'system calls tfidf_3gram-scaled', 
                ]
    malwares=["delay", "disorder", "freeze", "hop", "mimic", "noise", "repeat", "spoof"]
    device = 'pi4_4G'
    tw =  60
    resultsPath = os.getcwd() + '/'
    dataPath = resultsPath 
    clss = ["Isolation Forest", "One-Class SVM",  "Robust covariance","SGD One-Class SVM"]
    res = []
    for clsname in clss:
        for feature in features:  
            tsv_name = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, feature)
            encoded_trace_df = pd.read_csv(tsv_name, sep='\t')
            ft = [ast.literal_eval(i) for i in encoded_trace_df[feature]]
            encoded_trace_df[feature] = ft

            train_data = encoded_trace_df[encoded_trace_df.maltype=='normal'][feature].tolist()
            test_data = []
            for m in malwares:
                test_data +=  encoded_trace_df[encoded_trace_df.maltype==m][feature].tolist()

            clf, val_score, test_score, train_t, t_test = train(train_data, test_data, clsname)
            res.append((feature, clsname, val_score, test_score, train_t, t_test))
    
    
            clfName = '{}_{}.pk'.format(clsname, feature)    
            loc=open(resultsPath + clfName,'wb')
            pickle.dump(clf, loc)


    df = pd.DataFrame(res)
    df.columns = ['feature', 'clsname', 'val_score', 'test_score', 'train_t', 't_test']
    n = resultsPath + 'res.csv'
    df.to_csv(n)
 

if __name__ == "__main__":
    run()