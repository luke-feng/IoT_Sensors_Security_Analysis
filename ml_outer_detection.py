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

def train_models(feature, normal, clsname):
    X = normal[feature].tolist()
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
    print( feature + '_' + name)
    try:
        clf.fit(X_train)
        t2 =time.time()        
        y_pred = clf.predict(X_val)
        score = metrics.accuracy_score(y_val,y_pred)
    except:
        t2 =time.time()
        y_pred = []
        score = 0
    t = t2 -t1
    # print('Feature: {}, Model: {}, accuracy score: {}, training time is: {} seconds'.format(feature, name, score, t))
    return clf, y_pred, score, t

def test_models(encoded_trace_df, malware, feature, clf, clsname):
    dfs = encoded_trace_df[encoded_trace_df.maltype==malware]
    X_test = dfs[feature].tolist()
    y_test = np.ones(len(X_test))
    y_test = [-1  for i in range(0,len(y_test))]

    t1 =time.time()
    y_pred = clf.predict(X_test)
    t2 =time.time()
    score = metrics.accuracy_score(y_test, y_pred)
    t = t2 -t1
    # print('Feature: {}, Model: {}, accuracy score: {}, testing time is: {} seconds'.format(feature, clsname, score, t))
    return  y_pred, score, t

def run():
    feature = 'system calls frequency_1gram'
    clsname = 'Robust covariance'
    malwares=["delay", "disorder", "freeze", "hop", "mimic", "noise", "repeat", "spoof"]
    device = 'pi3'
    tw =  60
    resultsPath = os.getcwd() + '/'
    dataPath = resultsPath 


    tsv_name = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, feature)
    encoded_trace_df = pd.read_csv(tsv_name, sep='\t')
    ft = [ast.literal_eval(i) for i in encoded_trace_df[feature]]
    encoded_trace_df[feature] = ft
    preds = {}
    res = []

    normal = encoded_trace_df[encoded_trace_df.maltype=='normal']
    clf, y_pred, score, t = train_models(feature, normal, clsname)
    n = '{}_{}_{}'.format(feature, clsname, 'valid')    
    preds[n] = y_pred
    res.append((feature, clsname, 'normal', score, t))

    for malware in malwares:
        y_pred, score, t = test_models(encoded_trace_df, malware, feature, clf, clsname)
        n = '{}_{}_{}'.format(feature, clsname, malware)    
        preds[n] = y_pred
        res.append((feature, clsname, malware, score, t))
        
    clfName = '{}_{}.pk'.format(feature, clsname)    
    loc=open(resultsPath + clfName,'wb')
    pickle.dump(clf,loc)

    predName = 'preds_{}_{}.pk'.format(feature, clsname) 
    loc=open(resultsPath + predName,'wb')
    pickle.dump(preds,loc)   

    df = pd.DataFrame(res)
    df.columns = ['feature', 'clsname', 'malware', 'score', 't']
    n = resultsPath + '{}_{}.csv'.format(feature, clsname) 
    df.to_csv(n)
 

def test2(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    mem_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        mem_percents.append(p.memory_percent())
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents, mem_percents

if __name__ == "__main__":
    run()