# Algorithms complicity empiric estimation

import numpy as np
from scipy.optimize import curve_fit
from tabulate import tabulate


def log_model(n: int, a: float):
    return a * np.log2(n)


def linear_model(n: int, a: float):
    return a * n


def linear_log_model(n: int, a: float):
    return a * n * np.log2(n)


def quadratic_model(n: int, a: float):
    return a * n * n


def get_test_n_array(count: int):
    """
    count ϵ [1; 18]    
    """
    if (count < 0):
        count = 1
    elif (count > 18):
        count = 18

    res = []
    c = 1
    while (c <= count):
        res.append(np.power(10, c, dtype=np.int64))
        c = c + 1

    return res


def get_estimation(n_array: list[int], time_array: list[float], test_n_array: list[int]):
    (log_a), _ = curve_fit(log_model, n_array, time_array)
    (lin_a), _ = curve_fit(linear_model, n_array, time_array)
    (lin_log_a), _ = curve_fit(linear_log_model, n_array, time_array)
    (quadratic_a), _ = curve_fit(
        quadratic_model, n_array, time_array)

    res_list = []

    for N in test_n_array:
        res_list.append([N,
                         log_model(N, log_a),
                         linear_model(N, lin_a),
                         linear_log_model(N, lin_log_a),
                         quadratic_model(N, quadratic_a)])

    print(tabulate(res_list,
                   headers=[
                       'N',
                       'O(N)'
                       'O(Log2(N))',
                       'O(N * log2(N))',
                       'O(N^2)'],
                   tablefmt='orgtbl',
                   numalign="right"))


n_array = [100, 1000, 10000]
time_array = [0.063, 0.565, 5.946]
test_n_array = [8000, 16000, 32000, 64000, 128000]  # get_test_n_array(18)

get_estimation(n_array, time_array, test_n_array)
