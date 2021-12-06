import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import os,sys

def get_systemcall_name_with_fork(line):
	syscall = ''
	linelist = line.split(' ')
	pid = int(linelist[0])
	time_cost = re.split('<|>', linelist[-1])

	try:
		time_cost = float(time_cost[1])
	except:
		time_cost = 0
	for i,l in enumerate(linelist):
		if re.match(r'\d{2}:\d{2}:\d{2}', l) is not None:
			timestamp = l
			if '(' in linelist[i+1]:
				syscall = linelist[i+1].split('(')[0]
			elif '<...' in linelist[i+1]:
				syscall = linelist[i+2]
			elif '+++' in linelist[i+1]:
				syscall = linelist[i+2]
			elif '---' in linelist[i+1]:
				syscall = linelist[i+2]
			else:
				syscall = '!!'+linelist[i+1]
			break
	return [pid, timestamp, syscall, time_cost]

def get_systemcall_name_without_fork(line):
	syscall = ''
	linelist = line.split(' ')
	# pid = int(linelist[0])
	time_cost = re.split('<|>', linelist[-1])

	try:
		time_cost = float(time_cost[1])
	except:
		time_cost = 0
	for i,l in enumerate(linelist):
		if re.match(r'\d{2}:\d{2}:\d{2}', l) is not None:
			timestamp = l
			if '(' in linelist[i+1]:
				syscall = linelist[i+1].split('(')[0]
			elif '<...' in linelist[i+1]:
				syscall = linelist[i+2]
			elif '+++' in linelist[i+1]:
				syscall = linelist[i+2]
			elif '---' in linelist[i+1]:
				syscall = linelist[i+2]
			else:
				syscall = '!!'+linelist[i+1]
			break
	return [timestamp, syscall, time_cost]

def create_onehot_encoding(total, index):
	onehot = []
	for i in range(0, total):
		if i == index:
			onehot.append(1)
		else:
			onehot.append(0)
	return onehot

def create_vectorizers(corpus, ngram):
	syscall_dict = {}
	ngrams_dict = {}
	countvectorizer = CountVectorizer().fit(corpus)
	syscall_dict = countvectorizer.vocabulary_
	countvectorizer = CountVectorizer(ngram_range=(1, ngram)).fit(corpus)
	ngrams_dict = countvectorizer.vocabulary_
	tfidfvectorizer = TfidfVectorizer(ngram_range=(1, ngram), vocabulary=ngrams_dict).fit(corpus)
	hashingvectorizer = HashingVectorizer(n_features=2**7).fit(corpus)  
	return syscall_dict, ngrams_dict, countvectorizer, tfidfvectorizer, hashingvectorizer

def read_dict_from_file(dictfilepath):
	syscall_dict = dict()
	syscall_dict_onehot = dict()
	index_dict = dict()
	file_dict = pd.read_csv(dictfilepath,header=None)
	file_dict.columns = ['syscall', 'index']
	syscall = file_dict['syscall']
	index = file_dict['index']
	total = len(syscall)
	for i,sc in enumerate(syscall):
		syscall_dict[sc] = index[i]
		syscall_dict_onehot[sc] = create_onehot_encoding(total, index[i])
		index_dict[index[i]] = sc
	return syscall_dict,syscall_dict_onehot,index_dict

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

def get_distance(trace,head,tail):
	start = 0
	end = -1
	distance = 0.0
	for i,s in enumerate(trace):
		if s == head:
			start=i
			rest = trace[i+1:]
			# print(rest)
			if i+1 < len(trace):
				if rest.count(head)>0:
					end = rest.index(head)+start+1
					sort = trace[start+1:end]
					for j,t in enumerate(sort):
						if t==tail:
							distance += 1/(j+1)
				else:
					sort = trace[start+1:]
					for j,t in enumerate(sort):
						if t==tail:
							distance += 1/(j+1)
	return distance

def get_dependency_graph(trace,term_dict):
	dp = []
	for head in term_dict:
		dp_ = []
		for tail in term_dict:
			if head == tail:
				dp_.append(0)
			else:
				distance = get_distance(trace,head,tail)
				dp_.append(distance)
		dp.append(dp_)
	return dp

def get_dependency_graph(trace,term_dict):
	dp = []
	for head in term_dict:
		dp_ = []
		for tail in term_dict:
			if head == tail:
				dp_.append(0)
			else:
				distance = get_distance(trace,head,tail)
				dp_.append(distance)
		dp.append(dp_)
	return dp

def get_frequency_vector(trace, syscall_dict):
	syscall_frequency = []
	for syscall in syscall_dict:
		f = trace.count(syscall)
		syscall_frequency.append(f)
	return syscall_frequency

def get_bigram_dict(syscall_dict):
	bigram_dict = []
	for i in syscall_dict:
		for j in syscall_dict:
			bigram_dict.append((i,j))
	return bigram_dict

def get_trigram_dict(syscall_dict):
	trigram_dict = []
	for i in syscall_dict:
		for j in syscall_dict:
			for k in syscall_dict:
				trigram_dict.append((i,j,k))
	return trigram_dict

def get_ngram_trace(syscall_list, n):
	n_gram = list(nltk.ngrams(syscall_list, n))
	return n_gram

def from_trace_to_longstr(syscall_trace):
	tracestr = ''
	for syscall in syscall_trace:
		tracestr += syscall + ' '
	return tracestr

def from_raw_to_features(inputFilePath, syscall_dict, bigram_dict, syscall_dict_onehot):
	trace = []
	# turn raw data to standardlized syscalls trace
	with open(inputFilePath, 'r') as inputfile:
		for line in inputfile:
			syscall = get_systemcall_name_without_fork(line)
			if syscall[1].startswith('!!'):
				print(line)
			else:
				trace.append(syscall)	
	trace = pd.DataFrame(trace)
	trace.columns = ['timestamp', 'syscall', 'timecost']
   

	# get bi/tri-gram trace
	# syscall_trace = replace_with_UNK(trace, syscall_dict)
	bigram_trace = get_ngram_trace(syscall_trace, 2)
	trigram_trace = get_ngram_trace(syscall_trace, 3)
	tracestr = from_trace_to_longstr(syscall_trace)

	# get frequency features
	syscall_frequency = get_frequency_vector(syscall_trace, syscall_dict)
	bigranm_frequency = get_frequency_vector(bigram_trace, bigram_dict)
	# trigram_frequency = get_frequency_vector(trigram_trace, trigram_dict)

	# get onehot encoding
	syscall_one_hot =  trace_onehot_encoding(syscall_trace, syscall_dict_onehot)

	# get dependency graph
	dependency_graph = get_dependency_graph(syscall_trace,syscall_dict)

	return syscall_trace, tracestr, syscall_frequency, bigranm_frequency, syscall_one_hot, dependency_graph

def read_file(inputFilePath):
	trace = []
	# turn raw data to standardlized syscalls trace
	with open(inputFilePath, 'r') as inputfile:
		for line in inputfile:
			syscall = get_systemcall_name_without_fork(line)
			if syscall[1].startswith('!!'):
				print(line)
			else:
				trace.append(syscall)	
	trace = pd.DataFrame(trace)
	trace.columns = ['timestamp', 'syscall', 'timecost']
	tracestr = from_trace_to_longstr(trace['syscall'])
	return trace, tracestr

def read_all_rawdata(rootPath):
	filesName = os.listdir(rootPath)
	corpus_dataframe = []
	corpus = []
	for fn in filesName:
		inputFilePath = rootPath + fn
		trace, tracestr = read_file(inputFilePath)
		corpus_dataframe.append(trace)
		corpus.append(tracestr)
	return corpus_dataframe, corpus

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


def get_features():
	# data path
	rootPath ='/root/data/'
	#read files from local 
	corpus_dataframe, corpus = read_all_rawdata(rootPath)
	# create dicts 
	syscall_dict, ngrams_dict, countvectorizer, tfidfvectorizer, hashingvectorizer = create_vectorizers(corpus, 3)
	syscall_dict, syscall_dict_onehot = add_unk_to_dict(syscall_dict)

	# get features
	frequency_features = countvectorizer.transform(corpus)
	tfidf_features = tfidfvectorizer.transform(corpus)
	hashing_features = hashingvectorizer.transform(corpus)

	one_hot_features = []
	dependency_graph_features = []
	for trace in corpus_dataframe:
		one_hot = []
		dependency_graph = []
		syscall_trace = replace_with_unk(trace['syscall'].to_list(), syscall_dict)
		syscall_one_hot =  trace_onehot_encoding(syscall_trace, syscall_dict_onehot)
		dependency_graph = get_dependency_graph(syscall_trace,syscall_dict)
		one_hot_features.append(syscall_one_hot)
		dependency_graph_features.append(dependency_graph)

	# write features to the file
	timestamp = []
	for trace in corpus_dataframe:
	    t = trace['timestamp'][0]
	    timestamp.append(t)
	encoded_trace_df = pd.DataFrame([timestamp, corpus_dataframe,corpus,frequency_features.toarray() ,tfidf_features.toarray(),hashing_features.toarray(), dependency_graph_features, one_hot_features] ).transpose()
	encoded_trace_df.columns = ['timestamp', 'corpus_raw', 'corpus_str', 'frequency_features' ,'tfidf_features','hashing_features', 'dependency_graph_features', 'one_hot_features']
	encoded_trace_df.to_pickle('encoded_features.pkl')

get_features()