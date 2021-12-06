import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import os,sys
import tqdm
import pickle
import time
import numpy as np




def get_syscall_dict(ngrams_dict):
    syscall_dict = {}
    i = 0
    for ngram in ngrams_dict:
        if len(ngram.split()) == 1:
            syscall_dict[ngram] = i
            i+=1
    return syscall_dict

def create_vectorizers(corpus, ngram):
    syscall_dict = {}
    ngrams_dict = {}
    # countvectorizer = CountVectorizer().fit(corpus)
    # syscall_dict = countvectorizer.vocabulary_
    countvectorizer = CountVectorizer(ngram_range=(1, ngram)).fit(corpus)
    print('create count vectorizer finished')
    ngrams_dict = countvectorizer.vocabulary_
    syscall_dict = get_syscall_dict(ngrams_dict)
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1, ngram), vocabulary=ngrams_dict).fit(corpus)
    print('create tf-idf vectorizer finished')
    hashingvectorizer = HashingVectorizer(n_features=2**5).fit(corpus)  
    print('create hashing vectorizer finished')
    return syscall_dict, ngrams_dict, countvectorizer, tfidfvectorizer, hashingvectorizer


def from_trace_to_longstr(syscall_trace):
    tracestr = ''
    for syscall in syscall_trace:
        tracestr += syscall + ' '
    # print(tracestr)
    return tracestr


def read_all_rawdata( rawdataPath, rawFileNames):    
    corpus_dataframe, corpus = [],[]
    par = tqdm.tqdm(total=len(rawFileNames), ncols=100)
    for fn in rawFileNames:
        if '.csv' in fn:
            par.update(1)
            fp = rawdataPath + fn
            trace = pd.read_csv(fp)
            tr = trace['syscall'].tolist()             
            longstr = from_trace_to_longstr(tr)
            corpus_dataframe.append(trace)
            corpus.append(longstr)
    par.close()
    return corpus_dataframe, corpus


def create_onehot_encoding(total, index):
    onehot = []
    for i in range(0, total):
        if i == index:
            onehot.append(1)
        else:
            onehot.append(0)
    return onehot

def add_unk_to_dict(syscall_dict):
    total = len(syscall_dict)
    syscall_dict['unk'] = total
    syscall_dict_onehot = dict()
    for sc in syscall_dict:
        syscall_dict_onehot[sc] = create_onehot_encoding(total+1, syscall_dict[sc])
    return syscall_dict, syscall_dict_onehot


def replace_with_unk(syscall_trace, syscall_dict):
    for i, sc in enumerate(syscall_trace):
        if sc.lower() not in syscall_dict:
            syscall_trace[i] = 'unk'
    return syscall_trace

def trace_onehot_encoding(trace, syscall_dict_onehot):
    encoded_trace = []
    for syscall in trace:
        syscall = syscall.lower()
        if syscall.lower() in syscall_dict_onehot:
            one_hot = syscall_dict_onehot[syscall]
        else:
            syscall = 'UNK'
            one_hot = syscall_dict_onehot[syscall]
        encoded_trace.append(one_hot)
    return encoded_trace

def find_all_head(trace, head):
    starts, ends,se = [], [], []

    for i,s in enumerate(trace):
        if s == head:
            start=i
            starts.append(start)
            if len(starts) > 1:
                end = starts[-1] 
                ends.append(end)
        if i == len(trace)-1:
            end = len(trace)
            ends.append(end)
    se = [(starts[i], ends[i]) for i in range(0, len(starts))]
    return se


def get_dict_sequence(trace,term_dict):
    dict_sequence = []
    for syscall in trace:
        if syscall in term_dict:
            dict_sequence.append(term_dict[syscall])
        else:
            dict_sequence.append(term_dict['unk'])
    return dict_sequence


def read_dicts(dictPath):
    vectorizers = {}
    dicts ={}
    for i in range(1, 6):
        cvName = 'countvectorizer_ngram{}'.format(i)
        tvName = 'tfidfvectorizer_ngram{}'.format(i)
        hvName = 'hashingvectorizer_ngram{}'.format(i)
        ndName = 'ngrams_dict_ngram{}'.format(i)
        sdName = 'syscall_dict_ngram{}'.format(i)
        shdName = 'syscall_dict_onehot_ngram{}'.format(i)

        loc=open(dictPath + cvName+'.pk','rb')
        cv = pickle.load(loc)
        vectorizers[cvName] = cv

        loc=open(dictPath + tvName+'.pk','rb')
        tv = pickle.load(loc)
        vectorizers[tvName] = tv

        loc=open(dictPath + hvName+'.pk','rb')
        hv = pickle.load(loc)
        vectorizers[hvName] = hv

        loc=open(dictPath + ndName+'.pk','rb')
        nd = pickle.load(loc)
        dicts[ndName] = nd

        loc=open(dictPath + sdName+'.pk','rb')
        sd = pickle.load(loc)
        dicts[sdName] = sd

        loc=open(dictPath + shdName+'.pk','rb')
        shd = pickle.load(loc)
        dicts[shdName] = shd
    
    return vectorizers, dicts


def get_features():
    times = {} 
    base_dict_path = '/data/dict/'
    vectorizers, dicts = read_dicts(base_dict_path)
    print('get dicts finished!')
    features = []
    rawdataPath = '/data/60/'
    rawFileNames = os.listdir(rawdataPath)
    ids, maltype = [], []
    for fi in rawFileNames:
        if '.csv' in fi:
            fis = fi.split('_')
            fn = fis[0]
            i = '{}_{}_{}'.format(fis[0], fis[2], fis[3])
            maltype.append(fn)
            ids.append(i)

    features.append(ids)
    features.append(maltype)

    print('start to read rawdata')
    corpus_dataframe, corpus = read_all_rawdata( rawdataPath, rawFileNames)
    # loc=open(rawdataPath +'corpus_dataframe.pk','rb')
    # corpus_dataframe = pickle.load(loc)
    # loc=open(rawdataPath +'corpus.pk','rb')
    # corpus = pickle.load(loc)

    print('got rawdata')

    ndName = 'ngrams_dict_ngram{}'.format(1)
    sdName = 'syscall_dict_ngram{}'.format(1)
    shdName = 'syscall_dict_onehot_ngram{}'.format(1)
    nd = dicts[ndName]
    sd = dicts[sdName]
    shd = dicts[shdName]
    
    one_hot_features = []
    dict_sequence_features = []

    t1 = time.time()
    par = tqdm.tqdm(total=len(corpus_dataframe), ncols=100)
    for trace in corpus_dataframe:
        syscall_trace = replace_with_unk(trace['syscall'].to_list(), sd)
        syscall_one_hot =  trace_onehot_encoding(syscall_trace, shd)
        one_hot_features.append(syscall_one_hot)
        par.update(1)
    t2 = time.time()
    par.close()
    key = 'syscall_one_hot'
    t = t2 - t1
    times[key] = t
    print(key+": "+str(t))
    # pca_name = key+'_pca'
    # inputs = padding_onehot(one_hot_features, 160000)
    # one_hot_features_pca, pca = get_pca_feature(inputs)
    # pcas[pca_name] = pca
    features.append(one_hot_features)
    # features.append(one_hot_features_pca)
    par = tqdm.tqdm(total=len(corpus_dataframe), ncols=100)
    t1 = time.time()
    for trace in corpus_dataframe:
        syscall_trace = replace_with_unk(trace['syscall'].to_list(), sd)
        dict_sequence = get_dict_sequence(syscall_trace,sd)
        dict_sequence_features.append(dict_sequence)
        par.update(1)
    t2 = time.time()
    par.close()
    t = t2 - t1
    key = 'dict_sequence'
    times[key] = t
    print(key+": "+str(t))
    # pca_name = key+'_pca'
    # inputs = padding_dictencoding(dict_sequence_features, 160000)
    # dict_sequence_pca, pca = get_pca_feature(inputs)
    # pcas[pca_name] = pca
    features.append(dict_sequence_features)
    # features.append(dict_sequence_pca)

    for i in range(1, 6):
        cvName = 'countvectorizer_ngram{}'.format(i)
        tvName = 'tfidfvectorizer_ngram{}'.format(i)
        hvName = 'hashingvectorizer_ngram{}'.format(i)             

        cv = vectorizers[cvName]
        tv = vectorizers[tvName]
        hv = vectorizers[hvName]

        t1 = time.time()
        frequency_features = cv.transform(corpus)
        t2 = time.time()
        key = cvName
        t = t2 - t1
        times[key] = t
        print(key+": "+str(t))
        frequency_features = frequency_features.toarray()
        # frequency_pca_name = key+'_pca'
        # frequency_pca,pca = get_pca_feature(frequency_features)
        # pcas[frequency_pca_name] = pca               

        t1 = time.time()
        tfidf_features = tv.transform(corpus)
        t2 = time.time()
        t = t2 - t1
        key = tvName
        times[key] = t
        print(key+": "+str(t))
        tfidf_features = tfidf_features.toarray()
        # tfidf_pca_name = key+'_pca'
        # tfidf_pca,pca = get_pca_feature(tfidf_features)
        # pcas[tfidf_pca_name] = pca

        t1 = time.time()
        hashing_features = hv.transform(corpus)
        t2 = time.time()
        t = t2 - t1
        key = hvName
        times[key] = t
        print(key+": "+str(t))
        hashing_features = hashing_features.toarray()

        features.append(frequency_features)
        # features.append(frequency_pca)
        features.append(tfidf_features)
        # features.append(tfidf_pca)
        features.append(hashing_features)           
            
    encoded_trace_df = pd.DataFrame(features).transpose()
    encoded_trace_df.columns = ['ids', 'maltype','one hot encoding', 'dict index encoding', 
    'system calls frequency_1gram', 'system calls tfidf_1gram', 'system calls hashing_1gram',
    'system calls frequency_2gram', 'system calls tfidf_2gram', 'system calls hashing_2gram',
    'system calls frequency_3gram', 'system calls tfidf_3gram', 'system calls hashing_3gram',
    'system calls frequency_4gram', 'system calls tfidf_4gram', 'system calls hashing_4gram',
    'system calls frequency_5gram', 'system calls tfidf_5gram', 'system calls hashing_5gram'
    ]
    resultsPath = '/data/enc/'
    encoded_trace_df.to_pickle(resultsPath+'encoded_bow.pkl')  
    return times 

get_features()