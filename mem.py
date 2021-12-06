
# from memory_profiler import profile
# from line_profiler import LineProfiler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor



# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def test1():
    np.random.seed(42)

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate normal (not abnormal) training observations
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate new normal (not abnormal) observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    clf.fit(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

if __name__=='__main__':
    test1()