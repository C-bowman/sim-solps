from numpy import log, exp, ndarray


def gaussian_likelihood(data: ndarray, errors: ndarray, prediction: ndarray) -> float:
    z = (data - prediction) / errors
    return -0.5 * (z**2).sum()


def cauchy_likelihood(data: ndarray, errors: ndarray, prediction: ndarray) -> float:
    z = (data - prediction) / errors
    return -log(1 + z**2).sum()


def laplace_likelihood(data: ndarray, errors: ndarray, prediction: ndarray) -> float:
    z = (data - prediction) / errors
    return -abs(z).sum()


def logistic_likelihood(data: ndarray, errors: ndarray, prediction: ndarray) -> float:
    z = (prediction - data) / errors
    f = z + 2 * log(1 + exp(-z))
    return -f.sum()
