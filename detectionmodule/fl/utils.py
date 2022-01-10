from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
import openml
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
import pandas as pd
import ast



XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: SGDOneClassSVM) -> SGDOneClassSVM:
    """Returns the paramters of a sklearn OneClassSVM model."""
    if model.fit_intercept:
        params = (model.coef_, model.offset_)
    else:
        params = (model.coef_, model.offset_)
    return params


def set_model_params(
    model: SGDOneClassSVM, params: SGDOneClassSVM
) -> SGDOneClassSVM:
    """Sets the parameters of a sklean OneClassSVM model."""
    model.coef_ = params[0]
    model.offset_ = params[1]
    return model


def set_initial_params(model: OneClassSVM):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.OneClassSVM documentation for more
    information.
    """
    n_classes = 1  
    n_features = 100  # Number of features in dataset
    model.offset_=np.zeros(n_classes)
    # model.classes_ = np.array([i for i in range(10)])
    model.coef_ = np.zeros((n_classes, n_features))
    # if model.fit_intercept:
    #     model.intercept_ = np.zeros((n_classes,))


def load_all_data(feature):
    """read all data from csv file
    """
    dataPath = 'D:/mt_data/data/'
    filename = 'encoded_bow_{}.csv'.format(feature)
    t1 = dataPath+filename
    t1df = pd.read_csv(t1, sep='\t')
    t1data = [ast.literal_eval(i) for i in t1df['data']]
    data = pd.DataFrame([t1df['device'].tolist(), t1df['label'].tolist(), t1df['malware'].tolist(), t1data]).transpose()
    data.columns = ['device', 'label','malware', 'data']
    for d in t1data:
        d.pop(6)
        d.pop(0)
    return data

def load_data_node1(data, feature)-> Dataset:
    device ='pi3_t1'
    x_test= data[(data['device']==device)&(data['label']==-1)]['data'].tolist()
    x_train = data[(data['device']==device)&(data['label']==1)]['data'].tolist()
    pre_train = Nystroem(gamma=0.1, random_state=42)
    x_train = pre_train.fit_transform(x_train).tolist()
    x_test = pre_train.transform(x_test).tolist()
    y_train = [1 for i in range(0, len(x_train))]
    y_test = [-1 for i in range(0, len(x_test))]
    return (x_train, y_train), (x_test, y_test)


def load_data_node2(data, feature)-> Dataset:
    device ='pi3_t2'
    x_test= data[(data['device']=='pi3_t1')&(data['label']==-1)]['data'].tolist()
    x_train = data[(data['device']==device)&(data['label']==1)]['data'].tolist()
    pre_train = Nystroem(gamma=0.1, random_state=42)
    x_train = pre_train.fit_transform(x_train).tolist()
    x_test = pre_train.transform(x_test).tolist()
    y_train = [1 for i in range(0, len(x_train))]
    y_test = [-1 for i in range(0, len(x_test))]
    return (x_train, y_train), (x_test, y_test)


def load_data_node3(data, feature)-> Dataset:
    device ='pi4_2G_t1'
    x_test= data[(data['device']==device)&(data['label']==-1)]['data'].tolist()
    x_train = data[(data['device']==device)&(data['label']==1)]['data'].tolist()
    pre_train = Nystroem(gamma=0.1, random_state=42)
    x_train = pre_train.fit_transform(x_train).tolist()
    x_test = pre_train.transform(x_test).tolist()
    y_train = [1 for i in range(0, len(x_train))]
    y_test = [-1 for i in range(0, len(x_test))]
    return (x_train, y_train), (x_test, y_test)


def load_data_node4(data, feature)-> Dataset:
    device ='pi4_2G_t2'
    x_test= data[(data['device']=='pi4_2G_t1')&(data['label']==-1)]['data'].tolist()
    x_train = data[(data['device']==device)&(data['label']==1)]['data'].tolist()
    pre_train = Nystroem(gamma=0.1, random_state=42)
    x_train = pre_train.fit_transform(x_train).tolist()
    x_test = pre_train.transform(x_test).tolist()
    y_train = [1 for i in range(0, len(x_train))]
    y_test = [-1 for i in range(0, len(x_test))]
    return (x_train, y_train), (x_test, y_test)

def load_data_node5(data, feature)-> Dataset:
    device ='pi4_4G_t1'
    x_test= data[(data['device']==device)&(data['label']==-1)]['data'].tolist()
    x_train = data[(data['device']==device)&(data['label']==1)]['data'].tolist()
    pre_train = Nystroem(gamma=0.1, random_state=42)
    x_train = pre_train.fit_transform(x_train).tolist()
    x_test = pre_train.transform(x_test).tolist()
    y_train = [1 for i in range(0, len(x_train))]
    y_test = [-1 for i in range(0, len(x_test))]
    return (x_train, y_train), (x_test, y_test)


def load_data_node6(data, feature)-> Dataset:
    device ='pi4_4G_t2'
    x_test= data[(data['device']=='pi4_4G_t1')&(data['label']==-1)]['data'].tolist()
    x_train = data[(data['device']==device)&(data['label']==1)]['data'].tolist()
    pre_train = Nystroem(gamma=0.1, random_state=42)
    x_train = pre_train.fit_transform(x_train).tolist()
    x_test = pre_train.transform(x_test).tolist()
    y_train = [1 for i in range(0, len(x_train))]
    y_test = [-1 for i in range(0, len(x_test))]
    return (x_train, y_train), (x_test, y_test)


def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML.
    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )







# from typing import Tuple, Union, List
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# import openml

# XY = Tuple[np.ndarray, np.ndarray]
# Dataset = Tuple[XY, XY]
# LogRegParams = Union[XY, Tuple[np.ndarray]]
# XYList = List[XY]


# def get_model_parameters(model: LogisticRegression) -> LogRegParams:
#     """Returns the paramters of a sklearn LogisticRegression model."""
#     if model.fit_intercept:
#         params = (model.coef_, model.intercept_)
#     else:
#         params = (model.coef_,)
#     return params


# def set_model_params(
#     model: LogisticRegression, params: LogRegParams
# ) -> LogisticRegression:
#     """Sets the parameters of a sklean LogisticRegression model."""
#     model.coef_ = params[0]
#     if model.fit_intercept:
#         model.intercept_ = params[1]
#     return model


# def set_initial_params(model: LogisticRegression):
#     """Sets initial parameters as zeros Required since model params are
#     uninitialized until model.fit is called.
#     But server asks for initial parameters from clients at launch. Refer
#     to sklearn.linear_model.LogisticRegression documentation for more
#     information.
#     """
#     n_classes = 10  # MNIST has 10 classes
#     n_features = 784  # Number of features in dataset
#     model.classes_ = np.array([i for i in range(10)])

#     model.coef_ = np.zeros((n_classes, n_features))
#     if model.fit_intercept:
#         model.intercept_ = np.zeros((n_classes,))


# def load_mnist() -> Dataset:
#     """Loads the MNIST dataset using OpenML.
#     OpenML dataset link: https://www.openml.org/d/554
#     """
#     mnist_openml = openml.datasets.get_dataset(554)
#     Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
#     X = Xy[:, :-1]  # the last column contains labels
#     y = Xy[:, -1]
#     # First 60000 samples consist of the train set
#     x_train, y_train = X[:60000], y[:60000]
#     x_test, y_test = X[60000:], y[60000:]
#     return (x_train, y_train), (x_test, y_test)


# def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
#     """Shuffle X and y."""
#     rng = np.random.default_rng()
#     idx = rng.permutation(len(X))
#     return X[idx], y[idx]


# def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
#     """Split X and y into a number of partitions."""
#     return list(
#         zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    # )