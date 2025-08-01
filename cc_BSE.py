import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493


def bccd_t2_amps(mol:gto.Mole) -> tuple[np.ndarray,np.ndarray]:
    myhf = mol.HF.run() 
    mycc = cc.BCCD(myhf,conv_tol_normu=1e-8).run()

    print(mycc.e_tot)
    mo = mycc.mo_coeff

    print(f'Max. value in BCCD T1 amplitudes {abs(mycc.t1).max()}')
    print(f'Max. value in BCCD T2 amplitudes {abs(mycc.t2).max()}')

    t2 = mycc.t2

    # Get number of spatial orbitals
    n_occ = t2.shape[0]
    n_vir = t2.shape[2]

    t_ijab = t2
    print(t2.reshape(-1))
    t_ijba = -np.einsum("ijab->ijba", t2,optimize='optimal')

    t2_spin = np.zeros((n_occ*2,n_occ*2,n_vir*2,n_vir*2))
    t2_spin[:n_occ,:n_occ,:n_vir,:n_vir] = t_ijab-t_ijba
    t2_spin[n_occ:,n_occ:,n_vir:,n_vir:] = t_ijab-t_ijba
    t2_spin[:n_occ,n_occ:,:n_vir,n_vir:] = t_ijab
    t2_spin[n_occ:,:n_occ,n_vir:,:n_vir] = t_ijab
    t2_spin[n_occ:,:n_occ,:n_vir,n_vir:] = t_ijba
    t2_spin[:n_occ,n_occ:,n_vir:,:n_vir] = t_ijba

    # print(t2_spin)
    return mo, t2_spin
    
if __name__ == "__main__":

    # Define Molecule to calculate amplitudes and mo for
    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
                basis='cc-pVTZ',
                spin=0,
                symmetry=False,
                unit="Bohr")
    
    # get molecular orbitals and t2 amplitudes
    mo,t2 = bccd_t2_amps(mol)   
    core_spinorbs, vir_spinorbs = bse.get_spinorbs(mo)
    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    print(t2.shape)

    # Build electron repulsion integrals
    _, eri_ao = bse.spinor_one_and_two_e_int(mol)                       # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # antisymmetrised:
    oovv,ooov,vovv,ovvo,ovov = bse.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # n_occ x n_occ, n_vir x n_vir
    occ_selfeng, vir_selfeng = bse.get_self_energy(t2,oovv)

    # n_occ x n_occ, n_vir x n_vir
    fock_occ, fock_vir = bse.build_fock_matrices(mol)

    F_ij = occ_selfeng + fock_occ
    F_ab = vir_selfeng + fock_vir    

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')
    
    H_bse = F_abij - F_ijab + np.einsum("iabj->iajb", ovvo,optimize='optimal') + np.einsum("ikbc,jkca -> iajb", oovv,t2,optimize="optimal")

    H_bse = H_bse.reshape((n_occ*n_vir,n_occ*n_vir))
    e, v = np.linalg.eig(H_bse)

    np.savetxt("eigenvals.csv", e, delimiter=',')

