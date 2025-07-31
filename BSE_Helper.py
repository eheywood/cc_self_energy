r"""
Helper file for auxiliary codes
"""

import numpy as np
from scipy.linalg import block_diag


def spinor_one_and_two_e_int(mol):
    r"""
    Produce one- and two- electron integrals in the spinor basis
    :param mol:
    :param mo_basis:
    :return:
    """
    kin = mol.intor('int1e_kin')
    vnuc = mol.intor('int1e_nuc')
    eri = mol.intor('int2e')
    hcore = kin + vnuc

    spin_hcore = block_diag(hcore, hcore)
    n = hcore.shape[0]
    spin_eri = np.zeros((2*n,2*n,2*n,2*n))
    spin_eri[:n, :n, :n, :n] = eri
    spin_eri[n:, n:, n:, n:] = eri
    spin_eri[:n, :n, n:, n:] = eri
    spin_eri[n:, n:, :n, :n] = eri
    return np.real(spin_hcore), np.real(spin_eri)


def super_matrix_solver(A, B):
    r"""
    Solve RPA like equations
    :return: Eigenvalues, X, and Y matrices
    """
    nA = int(np.sqrt(A.size))
    nB = int(np.sqrt(B.size))
    A = A.reshape((nA, nA))
    B = B.reshape((nB, nB))

    # Construct supermatrix
    supermat = np.zeros((nA+nB, nA+nB))
    supermat[:nA, :nA] = A
    supermat[:nA, nA:] = B
    supermat[nA:, :nA] = -np.conj(B)
    supermat[nA:, nA:] = -np.conj(A)

    # Solve the eigenvalue problem
    e, v = np.linalg.eig(supermat)

    # In our current formulation, A and B are real matrices.
    # Eigenvalues of the supermatrix come in pairs.
    # We take the positive ones (assuming that they mean excitation energies)
    assert np.allclose(np.imag(e), np.zeros(e.shape), rtol=0, atol=1e-8)    # Real eigenvalues
    e = np.real(e)
    positive_idx = np.where(e > 0)[0]
    assert len(positive_idx) == len(e) // 2     # Half of the eigenvalues should be taken

    pos_e = e[positive_idx]
    pos_v = v[:, positive_idx]
    X = pos_v[:nA, :]
    Y = pos_v[nA:, :]
    return pos_e, X, Y

