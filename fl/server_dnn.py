import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn import metrics
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}

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

def train(net, train_loader, val_loader):    
    net = net.train() 
    loss_func = T.nn.MSELoss()
    optimizer = T.optim.Adam(net.parameters(), lr=0.01)
    max_epochs = 10
    # print("Starting training")
    last_val_loss = float("inf")
    patience = 0
    last_loss = 0
    for epoch in range(0, max_epochs):
        loss = 0
        # if epoch > 0 and epoch % (max_epochs/10) == 0:
        #     print("epoch = %6d" % epoch, end="")
        #     print(" prev total loss = %7.4f, perv total val-loss = %7.4f" %( last_loss,val_loss))
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
    # print("Training stop at epoch： %d" %epoch)
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
    loss = 0
    with T.no_grad():
        for i in range(0,len(X)):
            x = T.Tensor(X[i])
            y_t = net(x)
            t = loss_func(y_t,x).item()
            loss += t
            if t > down_threshold and t < up_threshold:
                y_pred.append(1)
            else:
                y_pred.append(-1)
    loss = loss/len(X)
    return y_pred, loss

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: T.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net

def get_eval_fn(net):
    """Return an evaluation function for server-side evaluation."""
    feature ='system calls frequency_1gram-scaled'
    data = utils.load_all_data(feature)
    (X_train, y_train), (X_test, y_test) = utils.load_data_node3(data, feature)
    # Split train val set 
    X_train = T.FloatTensor(X_train)

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        set_parameters(net, parameters)
        down_threshold, up_threshold = find_threshold(net, X_train)
        pred_val, loss = test(net, X_train, down_threshold, up_threshold)
        pred_test, loss = test(net, X_test, down_threshold, up_threshold)
        val_acc = metrics.accuracy_score(pred_val, y_train)
        test_acc = metrics.accuracy_score(pred_test, y_test)
        return loss, {"accuracy": (val_acc, test_acc)}
    return evaluate



if __name__ == "__main__":
    
     # Load model
    net = Autoencoder_DNN(15)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        fraction_fit=0.5,
        fraction_eval=0.5,
        eval_fn=get_eval_fn(net),
        on_fit_config_fn=fit_round,
    )

    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 3},
        strategy=strategy,
    )