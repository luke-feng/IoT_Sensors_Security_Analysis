from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os,sys
import pickle
import ast


dataPath = '/encoded/t1/'
dictPath = '/encoded/t1/'

devices = [ 'pi3', 'pi4_2G', 'pi4_4G']
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


def read_feature_tsv(device, tw, ftname):
    tsv_name = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, ftname)
    tsv_df = pd.read_csv(tsv_name, sep='\t')
    feature = [ast.literal_eval(i) for i in tsv_df[ftname]]
    tsv_df[ftname] = feature
    normal = tsv_df[tsv_df.maltype=='normal'][ftname].tolist() 
    X_train, X_val = train_test_split(normal, test_size=.3, shuffle=False) 
    return X_train, X_val, tsv_df

def normal(device, tw, ftname):
    X_train, X_val, tsv_df = read_feature_tsv(device, tw, ftname)
    scaler = StandardScaler().fit(X_train)
    data = scaler.transform(tsv_df[ftname].tolist())
    normaled_df = pd.DataFrame([tsv_df['ids'].tolist(), tsv_df['maltype'].tolist(), data.tolist()]).transpose()
    scaledName = dataPath+'{}_{}_{}-scaled.pk'.format(device, tw, ftname)
    loc = open(scaledName,'wb')
    pickle.dump(scaler, loc)
    loc.close()
    ftname = ftname+'-scaled'
    normaled_df.columns = ['ids', 'maltype', ftname]
    dfName = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, ftname)
    normaled_df.to_csv(dfName, sep='\t', index=None)


for device in devices:
    for ftname in features:
        normal(device, 60, ftname)
