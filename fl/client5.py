import warnings
import flwr as fl
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import utils
from sklearn import metrics

if __name__ == "__main__":
    # Load data from csv
    feature ='system calls frequency_1gram-scaled'
    data = utils.load_all_data(feature)
    (X_train, y_train), (X_test, y_test) = utils.load_data_node5(data, feature)
    # Split train val set 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.3, shuffle=False)
    # Create OneClassSVM Model
    model = SGDOneClassSVM(nu=0.05, shuffle=True, fit_intercept=True, random_state=42, tol=1e-6)
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class Client1(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train)
            # val_acc = metrics.accuracy_score(model.predict(X_val), y_val)
            # test_acc = metrics.accuracy_score(model.predict(X_test), y_test)
            # print(f"Training finished for round {config['rnd']}, accuracy {val_acc, test_acc}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_train+y_test, model.score_samples(X_train+X_test))
            val_acc = metrics.accuracy_score(model.predict(X_val), y_val)
            test_acc = metrics.accuracy_score(model.predict(X_test), y_test)
            print(f"accuracy {val_acc, test_acc}")
            return loss, len(X_test), {"accuracy": f"{val_acc, test_acc}"}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=Client1())