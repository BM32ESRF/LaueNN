# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:14:53 2022

@author: PURUSHOT
"""
import numpy as np
# ---------------------    Metric tensor
def ComputeMetricTensor(a, b, c, alpha, beta, gamma):
    r"""
    computes metric tensor G or G* from lattice parameters
    (either direct or reciprocal * ones)
    :param a,b,c,alpha,beta,gamma: lattice parameters (angles in degrees)
    :returns: 3x3 metric tensor
    """
    Alpha = alpha * DEG
    Beta = beta * DEG
    Gamma = gamma * DEG
    row1 = a * np.array([a, b * np.cos(Gamma), c * np.cos(Beta)])
    row2 = b * np.array([a * np.cos(Gamma), b, c * np.cos(Alpha)])
    row3 = c * np.array([a * np.cos(Beta), b * np.cos(Alpha), c])
    return np.linalg.inv(np.array([row1, row2, row3]))

def get_angular_distance(Gstar, two_hkl):
    hkl1 = np.copy(two_hkl)
    hkl2 = np.copy(two_hkl)
    # compute square matrix containing angles
    metrics = Gstar
    H1 = hkl1
    n1 = hkl1.shape[0]
    H2 = hkl2
    n2 = hkl2.shape[0]
    dstar_square_1 = np.diag(np.inner(np.inner(H1, metrics), H1))
    dstar_square_2 = np.diag(np.inner(np.inner(H2, metrics), H2))
    scalar_product = np.inner(np.inner(H1, metrics), H2) * 1.0
    d1 = np.sqrt(dstar_square_1.reshape((n1, 1))) * 1.0
    d2 = np.sqrt(dstar_square_2.reshape((n2, 1))) * 1.0
    outy = np.outer(d1, d2)
    ratio = scalar_product / outy
    ratio = np.round(ratio, decimals=7)
    tab_angulardist = np.arccos(ratio) / (np.pi / 180.0)
    np.putmask(tab_angulardist, np.abs(tab_angulardist) < 0.001, 400)
    return tab_angulardist

DEG = np.pi / 180.0
## lattice parameter of your material
a, b, c, alpha, beta, gamma = 6.839, 6.839, 14.08, 90.0, 90.0, 120.0
Gstar = ComputeMetricTensor(a, b, c, alpha, beta, gamma)

two_hkl = np.array([[2, -1, -9], [3, 0, 3]])
angular_distance = get_angular_distance(Gstar, two_hkl)

print("The angular distance between two hkl is ", angular_distance[0,1])

# # =============================================================================
# # Compute UVW from HKl (NOT SURE IF THIS WORKS PROPERLY)
# # =============================================================================
# def from_parameters(a, b, c, alpha, beta, gamma, x_aligned_with_a=True):
#     """
#     Create a Lattice using unit cell lengths and angles (in degrees).
#     :param float a: first lattice length parameter.
#     :param float b: second lattice length parameter.
#     :param float c: third lattice length parameter.
#     :param float alpha: first lattice angle parameter.
#     :param float beta: second lattice angle parameter.
#     :param float gamma: third lattice angle parameter.
#     :param bool x_aligned_with_a: flag to control the convention used to define the Cartesian frame.
#     """
#     alpha_r = np.radians(alpha)
#     beta_r = np.radians(beta)
#     gamma_r = np.radians(gamma)
#     if x_aligned_with_a:  # first lattice vector (a) is aligned with X
#         vector_a = a * np.array([1, 0, 0])
#         vector_b = b * np.array([np.cos(gamma_r), np.sin(gamma_r), 0])
#         c1 = c * np.cos(beta_r)
#         c2 = c * (np.cos(alpha_r) - np.cos(gamma_r) * np.cos(beta_r)) / np.sin(gamma_r)
#         vector_c = np.array([c1, c2, np.sqrt(c ** 2 - c1 ** 2 - c2 ** 2)])
#     else:  # third lattice vector (c) is aligned with Z
#         cos_gamma_star = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) / (np.sin(alpha_r) * np.sin(beta_r))
#         sin_gamma_star = np.sqrt(1 - cos_gamma_star ** 2)
#         vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
#         vector_b = [-b * np.sin(alpha_r) * cos_gamma_star, b * np.sin(alpha_r) * sin_gamma_star, b * np.cos(alpha_r)]
#         vector_c = [0.0, 0.0, float(c)]
#     return [vector_a, vector_b, vector_c]

# (h, k, l) = (2, -1, -9)
# matrix = from_parameters(a, b, c, alpha, beta, gamma)
# M = np.array(matrix, dtype=np.float64).reshape((3, 3))
# l_vect = M.dot(np.array([h, k, l]))
# ## rounding off Miller index
# n_vect = np.round(l_vect / np.linalg.norm(l_vect),4)

# min_vect = np.argsort(np.abs(n_vect))

# min_val = 0
# if n_vect[min_vect[0]] == 0:
#     if n_vect[min_vect[1]] == 0:
#         min_val = n_vect[min_vect[2]]
#     else:
#         min_val = n_vect[min_vect[1]]
# else:
#     min_val = n_vect[min_vect[0]]
# n_vect = np.round(n_vect / np.abs(min_val),0)
# print("UVW for a given hkl is ", n_vect)













