import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493
    
if __name__ == "__main__":

    # Define Molecule to calculate amplitudes and mo for
    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
                basis='cc-pVTZ',
                spin=0,
                symmetry=False,
                unit="Bohr")
    
    # get molecular orbitals and t2 amplitudes
    mo,t2,_,_ = bse.bccd_t2_amps(mol)   
    core_spinorbs, vir_spinorbs = bse.get_spinorbs(mo)
    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    # Build electron repulsion integrals
    _, eri_ao = bse.spinor_one_and_two_e_int(mol)                       # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # antisymmetrised:
    oovv,ooov,vovv,ovvo,ovov = bse.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # n_occ x n_occ, n_vir x n_vir
    occ_selfeng, vir_selfeng = bse.get_self_energy(t2,oovv)

    # n_occ x n_occ, n_vir x n_vir
    fock_occ, fock_vir = bse.build_fock_matrices(mol,n_occ,n_vir)

    F_ij = occ_selfeng + fock_occ
    F_ab = vir_selfeng + fock_vir    

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')
    
    H_bse = F_abij - F_ijab + np.einsum("iabj->iajb", ovvo,optimize='optimal') + np.einsum("ikbc,jkca -> iajb", oovv,t2,optimize="optimal")

    H_bse = H_bse.reshape((n_occ*n_vir,n_occ*n_vir))
    e, v = np.linalg.eig(H_bse)

    np.savetxt("eigenvals.csv", e, delimiter=',')

