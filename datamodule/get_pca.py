
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import ast

devices = [ 'pi4_2G', 'pi4_4G']
tw =  60

rootPath = '/'
dataPath = rootPath + 'encoded/t1/'

def read_feature_tsv(device, tw, ftname):
    tsv_name = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, ftname)
    tsv_df = pd.read_csv(tsv_name, sep='\t')
    feature = [ast.literal_eval(i) for i in tsv_df[ftname]]
    return feature, tsv_df


def get_pca(device, tw, ftname):
    feature, tsv_df = read_feature_tsv(device, tw, ftname)
    pca = PCA(n_components=100).fit(feature)
    pcaed_feature = pca.transform(feature)
    pcaed_df = pd.DataFrame([tsv_df['ids'].tolist(), tsv_df['maltype'].tolist(), pcaed_feature.tolist()]).transpose()
    pcaName = dataPath+'{}_{}_{}-pcas.pk'.format(device, tw, ftname)
    loc = open(pcaName,'wb')
    pickle.dump(pca, loc)
    loc.close()
    ftname = ftname+'-pcas'
    pcaed_df.columns = ['ids', 'maltype', ftname]
    dfName = dataPath+'encoded_bow{}_{}_{}.csv'.format(device, tw, ftname)
    pcaed_df.to_csv(dfName, sep='\t', index=None)

features = [ 
            'system calls frequency_2gram', 
            'system calls tfidf_2gram',
            'system calls frequency_3gram',
            'system calls tfidf_3gram', 
            'system calls frequency_4gram', 
            'system calls tfidf_4gram',
            'system calls frequency_5gram',
            'system calls tfidf_5gram'
            ]


for device in devices:
    for ftname in features:
        get_pca(device, tw, ftname)