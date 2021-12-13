import pandas as pd
import os,sys
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import ast
import multiprocessing as mp
import psutil
from numpy.core.fromnumeric import mean

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def read_feature_tsv(device, tw, ftname):
    dataPath = os.getcwd()+'/' +'onedata/'
    tsv_name = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, ftname)
    tsv_df = pd.read_csv(tsv_name)
    feature = [ast.literal_eval(i) for i in tsv_df[ftname]]
    tsv_df[ftname] = feature
    normal = tsv_df[ftname].tolist() 
    return normal

def normal(device, tw, ftname):
    normal = read_feature_tsv(device, tw, ftname)
    vecPath = os.getcwd()+'/' +'vectorizer/'
    scaledName = vecPath+'{}_{}_{}-scaled.pk'.format(device, tw, ftname)
    loc = open(scaledName,'rb')
    scaler = pickle.load(loc)
    t1 = time.time()
    data = scaler.transform(normal)
    t2 = time.time()
    t = t2-t1
    print(t)


def get_pca(device, tw, ftname):
    normal = read_feature_tsv(device, tw, ftname)
    vecPath = os.getcwd()+'/' +'vectorizer/'
    pcaName = vecPath+'{}_{}_{}-pcas.pk'.format(device, tw, ftname)
    loc = open(pcaName,'rb')
    pca = pickle.load(loc)
    t1 = time.time()
    data = pca.transform(normal)
    t2 = time.time()
    t = t2-t1
    print(t)

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
        time.sleep(0.001)

    worker_process.join()
    return cpu_percents, mem_percents

def main():
    device = 'pi3'
    tws = 60
    features = [ 
        'system calls frequency_1gram', 
        'system calls tfidf_1gram',
        'system calls frequency_2gram', 
        'system calls tfidf_2gram',
        'system calls frequency_3gram',
        'system calls tfidf_3gram', 
        'system calls frequency_4gram', 
        'system calls tfidf_4gram',
        'system calls frequency_5gram',
        'system calls tfidf_5gram'
        ]
    ftname = features[2]
    # normal(device, tws, ftname)
    print(ftname)
    get_pca(device, tws, ftname)

if __name__ == "__main__":
    cpu_percents, mem_percents = test2(target=main)
    cpu_percents = [i for i in cpu_percents if i !=0 ]
    mem_percents = [i for i in mem_percents if i !=0 ]
    print(mean(cpu_percents))
    print(mean(mem_percents)/1024/1024) 