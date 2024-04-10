import numpy as np
from scipy import sparse


class Hamiltonian:
    @staticmethod
    def make_spin_ops():
        ops = {}
        ops["Sz"] = sparse.diags([0.5, -0.5])
        sp = sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
        ops["S+"] = sp
        sm = sp.transpose()
        ops["S-"] = sm
        ops["Sx"] = (sp + sm) / 2.0
        ops["Sy"] = (sp - sm) / 2.0j
        ops["I"] = sparse.identity(2)
        return ops

    @staticmethod
    def create_single_site_operator_matrix(operator, i, num_sites):
        # 恒等演算子は2x2行列であるべきです
        identity_matrix = sparse.identity(2)
        # クロネッカー積の初期値を恒等演算子で設定する
        kron_product = sparse.identity(1)
        for position in range(num_sites):
            if position == i:
                kron_product = sparse.kron(kron_product, operator, "csr")
            else:
                kron_product = sparse.kron(kron_product, identity_matrix, "csr")
        return kron_product

    @staticmethod
    def create_two_site_operator_matrix(op_i, i, op_j, j, num_sites):
        if i >= num_sites or j >= num_sites or i < 0 or j < 0:
            raise ValueError(
                "Site indices must be within the range of the system size."
            )

        # kron_product = sparse.csr_matrix((1, 1))  # 初期化
        kron_product = sparse.identity(1)
        for position in range(num_sites):
            if position == i:
                current_operator = op_i
            elif position == j:
                current_operator = op_j
            else:
                current_operator = sparse.identity(2)

            kron_product = sparse.kron(kron_product, current_operator, "csr")

        operator_matrix = kron_product
        return operator_matrix
