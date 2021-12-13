import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import os,sys
import pickle
import time
import numpy as np
from numpy.core.fromnumeric import mean
import multiprocessing as mp
import psutil
# from sklearn.decomposition import PCA

def read_dicts(feature, base_dict_path, tw):
    vectorizers = {}
    dicts ={}
    dictPath = base_dict_path + '/'

    loc=open(dictPath + feature+ '.pk','rb')
    cv = pickle.load(loc)
    return cv

    # for i in range(1, 6):
    #     cvName = 'countvectorizer_ngram{}'.format(i)
    #     tvName = 'tfidfvectorizer_ngram{}'.format(i)
    #     hvName = 'hashingvectorizer_ngram{}'.format(i)
    #     ndName = 'ngrams_dict_ngram{}'.format(i)
    #     sdName = 'syscall_dict_ngram{}'.format(i)
    #     shdName = 'syscall_dict_onehot_ngram{}'.format(i)

    #     loc=open(dictPath + cvName+'.pk','rb')
    #     cv = pickle.load(loc)
    #     vectorizers[cvName] = cv

    #     loc=open(dictPath + tvName+'.pk','rb')
    #     tv = pickle.load(loc)
    #     vectorizers[tvName] = tv

    #     loc=open(dictPath + hvName+'.pk','rb')
    #     hv = pickle.load(loc)
    #     vectorizers[hvName] = hv

    #     loc=open(dictPath + ndName+'.pk','rb')
    #     nd = pickle.load(loc)
    #     dicts[ndName] = nd

    #     loc=open(dictPath + sdName+'.pk','rb')
    #     sd = pickle.load(loc)
    #     dicts[sdName] = sd

    #     loc=open(dictPath + shdName+'.pk','rb')
    #     shd = pickle.load(loc)
    #     dicts[shdName] = shd    
    # return vectorizers, dicts

def from_trace_to_longstr(syscall_trace):
    tracestr = ''
    for syscall in syscall_trace:
        tracestr += syscall + ' '
    # print(tracestr)
    return tracestr

def read_rawdata(rawdataPath):    
    trace = pd.read_csv(rawdataPath, header=None)
    tr = trace[2].tolist()             
    longstr = from_trace_to_longstr(tr)
    return longstr
    # return trace, longstr

def run():
    currentPath = os.getcwd()
    perd_datapath = currentPath+'/pred_data/'
    files = os.listdir(perd_datapath)
    base_dict_path = currentPath+'/vectorizer/'
    # vectorizers, _ = read_dicts(base_dict_path, 60)
    features = [ 
                'countvectorizer_ngram1','tfidfvectorizer_ngram1',
                'countvectorizer_ngram2','tfidfvectorizer_ngram2',
                'countvectorizer_ngram3','tfidfvectorizer_ngram3',
                'countvectorizer_ngram4','tfidfvectorizer_ngram4',
                'countvectorizer_ngram5','tfidfvectorizer_ngram5',
                'hashingvectorizer_ngram1'               
                ]
    feature = features[10]
    vec = read_dicts(feature,base_dict_path, 60)

    encoded_path = currentPath+'/encoded/'
    f = files[2]
    rawdataPath = perd_datapath + f
    longstr = read_rawdata(rawdataPath)
    features = vec.transform([longstr])


    # for vec in vectorizers:
    #     with open(encoded_path+vec+'.csv', 'w') as o:
    #         tot = 0
    #         for f in files:
    #             rawdataPath = perd_datapath + f
    #             trace, longstr = read_rawdata(rawdataPath)
    #             vectorizer = vectorizers[vec]
    #             t1 = time.time()
    #             features = vectorizer.transform([longstr]).toarray()
    #             t2 = time.time()
    #             t=t2-t1
    #             tot += t
    #             # print(features)
    #             # break
    #             o.write(str(features)+'\n')
    #         avt = tot/len(files)
    #         # break
    #         print(vec, avt)
    #     o.close()


def test2(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    mem_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        mem_percents.append(p.memory_full_info().rss)
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents, mem_percents

if __name__ == "__main__":
    cpu_percents, mem_percents = test2(target=run)
    print(mean(cpu_percents))
    print(mean(mem_percents)/1024/1024) 