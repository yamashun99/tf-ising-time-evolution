import numpy as np


def make_spin_ops():
    ops = {}
    ops["Sz"] = np.diag((0.5, -0.5))
    sp = np.zeros((2, 2))
    sp[0, 1] = 1
    ops["S+"] = sp
    sm = sp.transpose()
    ops["S-"] = sm
    ops["Sx"] = (sp + sm) / 2.0
    ops["Sy"] = (sp - sm) / 2.0j
    ops["I"] = np.identity(2)
    return ops


def create_single_site_operator_matrix(operator, i, num_sites):
    single_operator_matrix = np.zeros((2**num_sites, 2**num_sites))
    identity_matrix = np.identity(len(operator))
    kron_product = 1
    for position in range(num_sites):
        if position == i:
            kron_product = np.kron(kron_product, operator)
        else:
            kron_product = np.kron(kron_product, identity_matrix)
    single_operator_matrix += kron_product
    return single_operator_matrix


def create_two_site_operator_matrix(op_i, i, op_j, j, num_sites):
    if i >= num_sites or j >= num_sites or i < 0 or j < 0:
        raise ValueError("Site indices must be within the range of the system size.")

    operator_matrix = np.zeros((2**num_sites, 2**num_sites))
    identity_matrix = np.identity(max(len(op_i), len(op_j)))

    for position in range(num_sites):
        if position == i:
            current_operator = op_i
        elif position == j:
            current_operator = op_j
        else:
            current_operator = identity_matrix

        if position == 0:
            kron_product = current_operator
        else:
            kron_product = np.kron(kron_product, current_operator)

    operator_matrix += kron_product
    return operator_matrix
