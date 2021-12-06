import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM
from sklearn import metrics

import time
import re
import ast
import os,sys
import pickle

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def train_models(feature, normal):
    X = normal[feature].tolist()
    y = normal['maltype'].tolist()
    # mlb = LabelBinarizer()
    
    y = np.zeros(len(X))
    # h = .02  # step size in the mesh
    outliers_fraction = 0.15
    nu = 0.05
    results = []
    preds = []
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, random_state=42)
    y_val = [1 if i==0 else -1 for i in y_val]

    result = []
    pred = dict()
    classifiers = {
            "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
            "One-Class SVM": OneClassSVM(nu=outliers_fraction, kernel="rbf",gamma=0.1),
            "SGD One-Class SVM": SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4),
            "Isolation Forest": IsolationForest(contamination=outliers_fraction,random_state=42),
        }
    for name in classifiers:
        clf = classifiers[name]
        t1 =time.time()
        res = dict()
        clf.fit(X_train)
        t2 =time.time()
        y_pred = clf.predict(X_val)
        score = metrics.accuracy_score(y_val,y_pred)

        pred['valid_'+  feature + '_' + name] = y_pred
        t = t2 -t1
        res['Model'] ='valid_' + feature + '_' + name
        res['Accuracy'] = score
        res['Training time'] = t
        result.append(res)
        print('Model: {}, accuracy score: {}, training time is: {} seconds'.format(res['Model'], score, t))
    results.append(result)
    preds.append(pred)
    return classifiers, results, preds

def test_models(encoded_trace_df, malware, feature, classifiers):
    dfs = encoded_trace_df[encoded_trace_df.maltype==malware]
    X_test = dfs[feature].tolist()

    y_test = np.ones(len(X_test))
    y_test = [1 if i==0 else -1 for i in y_test]

    result = []
    pred = dict()    
    for name in classifiers:
        res = dict()
        clf = classifiers[name]
        t1 =time.time()
        y_pred = clf.predict(X_test)
        t2 =time.time()
        score = metrics.accuracy_score(y_test, y_pred)
        t = t2 -t1
        pred[malware +'_' + feature + '_' + name] = y_pred
        res['Model'] = malware +'_' + feature + '_' + name
        res['Accuracy'] = score
        res['Testing time'] = t
        result.append(res)
        print('Model: {}, accuracy score: {}, testing time is: {} seconds'.format( res['Model'], score, t))
    return  result, pred

def run(device, tw):

    resultsPath = 'f:/temp/' 
    dataPath = 'f:/temp/'   
    resultsdict = dict()
    predsdict = dict()
    classifiersdict = dict()

    for feature in features:
        #train stage
        # read data from file
        tsv_name = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, feature)
        encoded_trace_df = pd.read_csv(tsv_name, sep='\t')
        ft = [ast.literal_eval(i) for i in encoded_trace_df[feature]]
        encoded_trace_df[feature] = ft

        normal = encoded_trace_df[encoded_trace_df.maltype=='normal']
        clfs, results, preds = train_models(feature, normal)
        
        resultsdict[feature+'_validation'] = results
        predsdict[feature+'_validation'] = preds
        classifiersdict[feature] = clfs
 
        # testing stage
        for malware in malwares:
            results, preds = test_models(encoded_trace_df, malware, feature, clfs)
            resultsdict[malware +'_' + feature] = results
            predsdict[malware +'_' + feature] = preds
            
    loc=open(resultsPath+'classifiers.pk','wb')
    pickle.dump(classifiersdict,loc)
    loc=open(resultsPath+'results.pk','wb')
    pickle.dump(resultsdict,loc)
    loc=open(resultsPath+'preds.pk','wb')
    pickle.dump(predsdict,loc)   
 
    rd = []
    for rs in resultsdict:
        for r in resultsdict[rs]:
            for s in r:
                rd.append(s)

    rd = pd.DataFrame(rd)
    md = [i.split('_') for i in rd['Model']]
    md = pd.DataFrame(md)
    md.columns  = ['Dataset','Features','Ngram','Model']
    nrd=pd.DataFrame([md['Dataset'],md['Features'],md['Ngram'], md['Model'], rd['Accuracy']]).transpose()
    nrd.to_csv(resultsPath+'results.csv',index=None)

if __name__ == "__main__":
    features = [#'one hot encoding', 'dict index encoding',
                # 'system calls dependency graph', 
                'system calls frequency_1gram',
                'system calls tfidf_1gram', 
                'system calls hashing_1gram',
                'system calls frequency_2gram', 
                'system calls tfidf_2gram',
                'system calls hashing_2gram',
                'system calls frequency_3gram',
                'system calls tfidf_3gram',
                'system calls hashing_3gram',
                # 'system calls frequency_4gram', 
                # 'system calls tfidf_4gram',
                # 'system calls hashing_4gram', 
                # 'system calls frequency_5gram',
                # 'system calls tfidf_5gram', 
                # 'system calls hashing_5gram',
                'system calls frequency_2gram-pcas', 
                'system calls tfidf_2gram-pcas',
                'system calls frequency_3gram-pcas',
                'system calls tfidf_3gram-pcas', 
                'system calls frequency_4gram-pcas', 
                'system calls tfidf_4gram-pcas',
                'system calls frequency_5gram-pcas',
                'system calls tfidf_5gram-pcas'
                ]

    malwares=["delay", "disorder", "freeze", "hop", "mimic", "noise", "repeat", "spoof"]
    device = 'pi3'
    tw =  60
    run(device, tw)