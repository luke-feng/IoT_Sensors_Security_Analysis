import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import time
import multiprocessing as mp
import psutil
import numpy as np

def test1():
    np.random.seed(42)

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate normal (not abnormal) training observations
    X = 0.3 * np.random.randn(10000, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate new normal (not abnormal) observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)

    for _ in range(1):
        clf.fit(X_train)
        time.sleep(0.05)
    # clf.fit(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


def test2(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents

if __name__=='__main__':
    cpu_percents = test2(target=test1)
    print(cpu_percents)
