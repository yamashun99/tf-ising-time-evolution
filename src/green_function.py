import numpy as np
from make_hamiltonian_dense import *
from make_hamiltonian import *


def make_dynamical_structure_factor(eigval, eigvec, S0, S1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (
            (eigvec[:, 0].conj() @ S0 @ eigvec[:, j])
            * (eigvec[:, j].conj() @ S1 @ eigvec[:, 0])
            / (E0 - Ej + omega + 0.1j)
        )
    return G


def make_Gij(eigval, eigvec, L, i, j, omega):
    sp_ops = make_spin_ops()
    sz = sp_ops["Sz"]
    sp = sp_ops["S+"]
    sm = sp_ops["S-"]
    s0 = sp_ops["I"]
    szi = [s0] * i + [sz] + [s0] * (L - i - 1)
    szj = [s0] * j + [sz] + [s0] * (L - j - 1)
    Si = make_matrix(szi)
    Sj = make_matrix(szj)
    G = make_dynamical_structure_factor(eigval, eigvec, Sj, Si, omega)
    return G


def calculate_total_correlation(eigenvalues, eigenvectors, num_sites, operator):
    ground_state_energy = eigenvalues[0]

    degenerate_indices = np.where(
        np.isclose(eigenvalues, ground_state_energy, atol=1e-10)
    )[0]
    total_ground_state = eigenvectors[:, degenerate_indices].sum(axis=1)

    correlations = []
    for site_index in range(1, num_sites):  # 0を含めてすべてのサイトに対してループ
        sitei_operator = Hamiltonian.create_single_site_operator_matrix(
            operator, site_index, num_sites
        )
        two_site_operator = Hamiltonian.create_two_site_operator_matrix(
            operator, 0, operator, site_index, num_sites
        )

        # 総和状態での期待値計算
        exp_value_two_site = (
            total_ground_state.T @ two_site_operator @ total_ground_state
        )
        exp_value_site0 = (
            total_ground_state.T @ sitei_operator @ total_ground_state
        )  # サイト0との一体演算子の期待値
        exp_value_sitei = exp_value_site0  # 自己相関のため、これは同じ

        # 自己相関を除くために条件を設定
        if site_index != 0:
            exp_value_sitei = total_ground_state.T @ sitei_operator @ total_ground_state

        # 相関を計算
        correlation = exp_value_two_site - exp_value_site0 * exp_value_sitei
        correlations.append(correlation)

    return correlations
