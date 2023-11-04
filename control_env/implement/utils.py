import numpy as np


def build_a_matrix(a_table):
    matrix_a = np.zeros([len(a_table), len(a_table)])

    for i, item in enumerate(a_table):
        for j in item:
            matrix_a[i, j] = 1
    return matrix_a


def build_d_matrix(a_table):
    matrix_d = np.zeros([len(a_table), len(a_table)])
    for i, item in enumerate(a_table):
        matrix_d[i, i] = len(item)
    return matrix_d


def build_b_matrix(link_table):
    matrix_b = np.zeros([len(link_table), len(link_table)])
    for item in link_table:
        matrix_b[item, item] = 1
    return matrix_b
