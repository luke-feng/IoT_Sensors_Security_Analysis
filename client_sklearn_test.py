import warnings
import flwr as fl
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from random import randint
from sklearn import metrics
import ast
from sklearn.model_selection import train_test_split
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY, XY]
ModelParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]
Model = SGDOneClassSVM

def get_model_parameters(model: Model) -> ModelParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    coef_ = model.coef_
    # intercept_= model.intercept_
    params = (coef_, )
    # if model.fit_intercept:
    #     params = (model.coef_, model.intercept_)
    # else:
    #     params = (model.coef_,)
    return params


def set_model_params(
    model: Model, params: ModelParams
) -> Model:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    # model.intercept_ = params[1]
    # if model.fit_intercept:
    #     model.intercept_ = params[1]
    return model


def set_initial_params(model: Model, n_features):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 1  # MNIST has 10 classes
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    model.offset_ = np.zeros(1)
    # if model.fit_intercept:
    #     model.intercept_ = np.zeros((n_classes,))


def load_dataset(device, tw, feature) -> Dataset:
    """Loads the MNIST dataset from local csv.
    """

    tsv_name = 'encoded_bow{}_{}_{}.csv'.format(device, tw, feature)
    encoded_trace_df = pd.read_csv(tsv_name, sep='\t')
    ft = [ast.literal_eval(i) for i in encoded_trace_df[feature]]
    encoded_trace_df[feature] = ft

    normal = encoded_trace_df[encoded_trace_df.maltype=='normal']
    abnoraml = encoded_trace_df[encoded_trace_df.maltype!='normal']
    X = normal[feature].tolist()
    y = [1 for i in range(0,len(X))]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.3, random_state=42)
    X_test = abnoraml[feature].tolist()
    y_test = [-1 for i in range(0,len(X_test))]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    device = 'pi3'
    tw = 60
    feature = 'system calls frequency_1gram'
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(device, tw, feature)
    n_features = len(X_train[0])

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)

    # Create LogisticRegression Model
    model = SGDOneClassSVM(nu=0.05, shuffle=True, fit_intercept=True, random_state=42, tol=1e-4)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            print("start fitting")
            set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train)
            print(f"Training finished for round {config['rnd']}")
            return get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            print("start evaluating")
            set_model_params(model, parameters)
            # loss = log_loss(y_test, model.score_samples(X_test))
            y_pred_val = model.predict(X_val)
            sco_val = model.score_samples(X_val)
            loss_val = log_loss(y_val, sco_val)
            accuracy_val = metrics.accuracy_score(y_val, y_pred_val)

            y_pred_test = model.predict(X_test)
            sco_test = model.score_samples(X_test)
            loss_test = log_loss(y_test, sco_test)
            accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
            # accuracy = model.score(X_test, y_test)
            print(f"evaluation finished for round {config['rnd']}, validation accuracy is {accuracy_val}")
            return loss_test, len(X_test), {"accuracy_validation": accuracy_val, "test_validation": accuracy_test}

    # Start Flower client
    fl.client.start_numpy_client("192.168.1.105:8080", client=MnistClient())






