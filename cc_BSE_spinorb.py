import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def build_gfock(t2, oovv, n_occ,n_vir) -> tuple[np.ndarray,np.ndarray]:
    # n_occ x n_occ, n_vir x n_vir
    occ_selfeng, vir_selfeng = bse.get_self_energy(t2,oovv)

    # n_occ x n_occ, n_vir x n_vir
    fock_occ, fock_vir = bse.build_fock_matrices(mol,n_occ,n_vir) 

    F_ij = occ_selfeng + fock_occ
    F_ab = vir_selfeng + fock_vir

    return F_ij,F_ab

def build_bse(F_ij:np.ndarray,F_ab,ovvo,oovv,t2, n_occ, n_vir) -> np.ndarray:
    
    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir))
    
    H_bse = F_abij - F_ijab + np.einsum("iabj->iajb", ovvo,optimize='optimal') + np.einsum("ikbc, jkca ->iajb",oovv, t2, optimize='optimal')
    
    return H_bse

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
    _, eri_ao = bse.spinor_one_and_two_e_int(mol)                   # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # antisymmetrised:
    oovv,ooov,vovv,ovvo,ovov = bse.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)
    # goovv = np.einsum("pi,qa,pqrs,rj,sb->iajb", core_spinorbs, core_spinorbs, eri_ao, vir_spinorbs, vir_spinorbs, optimize="optimal")

    # n_occ x n_occ, n_vir x n_vir
    # occ_selfeng, vir_selfeng = bse.get_self_energy(t2,oovv)

    # n_occ x n_occ, n_vir x n_vir
    # fock_occ, fock_vir = bse.build_fock_matrices(mol,n_occ,n_vir)

    # n_occ x n_occ, n_vir x n_vir
    gfock_occ, gfock_vir = build_gfock(t2, oovv, n_occ,n_vir)

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    hbse = build_bse(gfock_occ,gfock_vir,ovvo,oovv,t2, n_occ, n_vir)
    hbse = hbse.reshape((n_occ*n_vir,n_occ*n_vir))
    
    e, v = np.linalg.eig(hbse)
    print(e)

    np.savetxt("eigenvals_spin.csv", np.sort(np.real(e)), delimiter=',')

