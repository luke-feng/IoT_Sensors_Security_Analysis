import os
import pandas as pd
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
res = []

clss = ["Isolation Forest", "One-Class SVM",  "Robust covariance","SGD One-Class SVM"]
rootpath = os.getcwd() + '/'
for c in clss:
	for f in features:
		fn = rootpath + '{}_{}.pk'.format(c, f)
		fsize = os.path.getsize(fn)
		fsize = fsize/float(1024)
		res.append((c,f,fsize))
		print(fn, fsize)
df = pd.DataFrame(res)
df.columns = ['clsname', 'features', 'fsize']

n = rootpath + 'fsize.csv'
df.to_csv(n)



