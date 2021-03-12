import numpy as np

def generate_cardinality(N, p = 2):
    return [(x/ N)**p for x in np.arange(N, 0, -1)]


def sugeno_fuzzy_integral(X, measure=None, axis = 0, keepdims=True):
    if measure is None:
        measure = generate_cardinality(X.shape[axis])

    return sugeno_fuzzy_integral_generalized(X, measure, axis, np.minimum, np.amax, keepdims)


def sugeno_fuzzy_integral_generalized(X, measure, axis = 0, f1 = np.minimum, f2 = np.amax, keepdims=True):
    X_sorted = np.sort(X, axis = axis)
    return f2(f1(np.take(X_sorted, np.arange(0, X_sorted.shape[axis]), axis), measure), axis=axis, keepdims=keepdims)
