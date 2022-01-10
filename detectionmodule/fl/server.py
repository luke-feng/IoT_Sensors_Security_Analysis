import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn import metrics
from typing import Dict, List, Tuple, Optional
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: SGDOneClassSVM):
    """Return an evaluation function for server-side evaluation."""
    feature ='system calls frequency_1gram-scaled'
    data = utils.load_all_data(feature)
    (X_train, y_train), (X_test, y_test) = utils.load_data_node1(data, feature)

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_train+y_test, model.score_samples(X_train+X_test))
        val_acc = metrics.accuracy_score(model.predict(X_train), y_train)
        test_acc = metrics.accuracy_score(model.predict(X_test), y_test)
        return loss, {"accuracy": (val_acc, test_acc)}
    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = SGDOneClassSVM(nu=0.05, shuffle=True, fit_intercept=True, random_state=42, tol=1e-6)
    utils.set_initial_params(model)
    strategy =  fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )

    fl.server.start_server("localhost:8080", strategy=strategy, config={"num_rounds": 5})

# import flwr as fl
# import utils
# from sklearn.metrics import log_loss
# from sklearn.linear_model import LogisticRegression
# from typing import Dict


# def fit_round(rnd: int) -> Dict:
#     """Send round number to client."""
#     return {"rnd": rnd}


# def get_eval_fn(model: LogisticRegression):
#     """Return an evaluation function for server-side evaluation."""

#     # Load test data here to avoid the overhead of doing it in `evaluate` itself
#     _, (X_test, y_test) = utils.load_mnist()

#     # The `evaluate` function will be called after every round
#     def evaluate(parameters: fl.common.Weights):
#         # Update model with the latest parameters
#         utils.set_model_params(model, parameters)
#         loss = log_loss(y_test, model.predict_proba(X_test))
#         accuracy = model.score(X_test, y_test)
#         return loss, {"accuracy": accuracy}

#     return evaluate


# # Start Flower server for five rounds of federated learning
# if __name__ == "__main__":
#     model = LogisticRegression()
#     utils.set_initial_params(model)
#     strategy = fl.server.strategy.FedAvg(
#         min_available_clients=2,
#         eval_fn=get_eval_fn(model),
#         on_fit_config_fn=fit_round,
#     )
#     fl.server.start_server("localhost:8080", strategy=strategy, config={"num_rounds": 5})