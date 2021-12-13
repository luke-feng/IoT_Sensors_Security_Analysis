import pandas as pd
import os,sys
import pickle
import copy
import torch as T
import torch.utils.data as data
import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn import metrics
import ast

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class Autoencoder_DNN(T.nn.Module):  # input_len-32-8-32-input_len
  def __init__(self, input_len):
    super(Autoencoder_DNN, self).__init__()
    self.fc1 = T.nn.Linear(input_len, 64)
    self.fc2 = T.nn.Linear(64, 16)
    self.fc3 = T.nn.Linear(16, 8)
    self.fc4 = T.nn.Linear(8, 16)
    self.fc5 = T.nn.Linear(16, 64)
    self.fc6 = T.nn.Linear(64, input_len)

  def encode(self, x):  # input_len-32-8
    z = T.relu(self.fc1(x))
    z = T.relu(self.fc2(z)) 
    z = T.relu(self.fc3(z)) 
    return z  

  def decode(self, x):  # 8-32-input_len
    z = T.relu(self.fc4(x))
    z = T.relu(self.fc5(z)) 
    z = T.relu(self.fc6(z)) 
    return z
    
  def forward(self, x):  # 65-32-8-32-65
    z = self.encode(x) 
    z = self.decode(z) 
    return z  # in [0.0, 1.0]

def train(train_loader, val_loader, input_len):
    net = Autoencoder_DNN(input_len)
    net = net.train() 
    loss_func = T.nn.MSELoss()
    optimizer = T.optim.Adam(net.parameters(), lr=0.01)
    max_epochs = 100
    print("Starting training")
    last_val_loss = float("inf")
    patience = 0
    last_loss = 0
    for epoch in range(0, max_epochs):
        loss = 0
        if epoch > 0 and epoch % (max_epochs/10) == 0:
            print("epoch = %6d" % epoch, end="")
            print(" prev total loss = %7.4f, perv total val-loss = %7.4f" %( last_loss,val_loss))
        for curr_bat in train_loader:
            X = T.Tensor(curr_bat)
            optimizer.zero_grad()
            oupt = net(X)
            loss_obj = loss_func(oupt, X)  # note X not Y
            loss += loss_obj.item()
            loss_obj.backward()
            optimizer.step()
        last_loss = loss
        val_loss = 0
        with T.no_grad():
            for curr_bat in val_loader:
                X = T.Tensor(curr_bat)
                oupt = net(X)
                val_loss_obj = loss_func(oupt, X)  # note X not Y
                val_loss += val_loss_obj.item()
            # print(loss, val_loss)
            if val_loss < last_val_loss:
                last_val_loss = val_loss
                patience = 0
            else:
                patience += 1
        if patience >= 10:
            break               
    print("Training stop at epoch： %d" %epoch)
    return net

def find_threshold(net, X_train):
    net = net.eval()
    loss_func = T.nn.MSELoss()
    with T.no_grad():
        x_t = T.Tensor(X_train)
        y_t = net(x_t)
        y_pred = np.array([loss_func(y_t[i],x_t[i]).item() for i in range(0,len(x_t))])
    down_threshold = np.percentile(y_pred, 2.5)
    up_threshold = np.percentile(y_pred, 97.5)
    return down_threshold, up_threshold

def test(net, X, down_threshold, up_threshold):
    net = net.eval()  
    loss_func = T.nn.MSELoss()
    y_pred = []
    with T.no_grad():
        for i in range(0,len(X)):
            x = T.Tensor(X[i])
            y_t = net(x)
            t = loss_func(y_t,x).item()
            if t > down_threshold and t < up_threshold:
                y_pred.append(1)
            else:
                y_pred.append(-1)
    return y_pred


def run(train_data, test_data):
    X = train_data
    y = [1 for i in range(0,len(X))]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, shuffle=False)
    y_val = [1 for i in range(0,len(X_val))]

    X_train = T.FloatTensor(X_train)
    X_val = T.FloatTensor(X_val)
    train_loader = data.DataLoader(X_train, batch_size=256, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)
    val_loader = data.DataLoader(X_val, batch_size=256, shuffle=False, drop_last=False, num_workers=2)
    input_len = len(X_train[0])
    
    t1 =time.time()
    net = train(train_loader,val_loader, input_len)
    t2 =time.time()   
    down_threshold, up_threshold = find_threshold(net, X_train)
    y_pred = test(net, X_val,down_threshold, up_threshold)   
    val_score = metrics.accuracy_score(y_val,y_pred)

    train_t = t2 -t1

    X_test = test_data
    y_test = [-1  for i in range(0,len(X_test))]
    x_t1 = [X_test[0]]
	x_t1 = T.FloatTensor(x_t1)
    X_test = T.FloatTensor(X_test)
    t1 =time.time()
    y_pred = y_pred = test(net, x_t1, down_threshold, up_threshold)
    t2 =time.time()
    t_test = t2 -t1

    y_pred = y_pred = test(net, X_test, down_threshold, up_threshold)
    test_score = metrics.accuracy_score(y_test, y_pred)

    return net, val_score, test_score, train_t, t_test

def main():
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
    device = 'pi4_2G'
    tw =  60
    resultsPath = os.getcwd() + '/'
    dataPath = resultsPath 
    clss = ["Autoencoder"]
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

            clf, val_score, test_score, train_t, t_test = run(train_data, test_data)
            res.append((feature, clsname, val_score, test_score, train_t, t_test))
    
    
            clfName = '{}_{}.pk'.format(clsname, feature)    
            loc=open(resultsPath + clfName,'wb')
            pickle.dump(clf, loc)


    df = pd.DataFrame(res)
    df.columns = ['feature', 'clsname', 'val_score', 'test_score', 'train_t', 't_test']
    n = resultsPath + 'res_dnn.csv'.format(feature, clsname) 
    df.to_csv(n)

if __name__ == "__main__":
    main()