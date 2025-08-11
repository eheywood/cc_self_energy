import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def get_selfenergy_spatial(t2,oovv, goovv):
    selfener_occ_1 = np.einsum('ikab, jkab -> ij', oovv, t2, optimize="optimal")
    selfener_occ_2 = -np.einsum('ikab, jkba -> ij', oovv, t2, optimize="optimal")
    selfener_occ_3 = np.einsum('ikab, jkab -> ij', goovv, t2, optimize="optimal")
    selfener_occ_4 = np.einsum('ikba, jkab -> ij', goovv, t2, optimize="optimal")

    selfener_occ = 0.5*(selfener_occ_1 + selfener_occ_2 + selfener_occ_3 + selfener_occ_4)
    
    selfener_vir_1 = np.einsum('ijbc, ijac -> ab', oovv, t2, optimize="optimal")
    selfener_vir_2 = - np.einsum('ijbc, ijca -> ab', oovv, t2, optimize="optimal")
    selfener_vir_3 = np.einsum('ijbc, ijac -> ab', goovv, t2, optimize="optimal")
    selfener_vir_4 = np.einsum('ijcb, ijac -> ab', goovv, t2, optimize="optimal")

    selfener_vir = -0.5*(selfener_vir_1 + selfener_vir_2 + selfener_vir_3 + selfener_vir_4)
    return selfener_occ, selfener_vir


def build_bse(F_ij, F_ab, n_occ, n_vir, goovv, ovvo, govov, t2) -> np.ndarray:

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    nspincase = 2
    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir,nspincase))
    
    for i in range(2):
        H_bse[:,:,:,:,i] = F_abij - F_ijab

        if i == 0:
            # ispincase = 1  or 2

            # ovvo term
            H_bse[:,:,:,:,i] += np.einsum("iabj->iajb", ovvo, optimize="optimal")

            # contraction term
            term1 = np.einsum("ikbc, jkca -> iajb", oovv,t2,optimize="optimal")
            term2 = -np.einsum("ikbc, jkac -> iajb", oovv,t2,optimize="optimal")
            term3 = - np.einsum("ikbc, jkac -> iajb", goovv,t2,optimize="optimal")
            
            H_bse[:,:,:,:,i] += term1 + term2 + term3

        else:
            # ispincase = 3 or 4

            # govov term
            H_bse[:,:,:,:,i] += -govov

            # contraction term
            H_bse[:,:,:,:,i] += - np.einsum("ikcb, jkca -> iajb", goovv, t2,optimize="optimal") 

    return H_bse

if __name__ == "__main__":

    # Define Molecule to calculate amplitudes and mo for
    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
                basis='cc-pVTZ',
                spin=0,
                symmetry=False,
                unit="Bohr")

    
    # get molecular orbitals and t2 amplitudes
    myhf = mol.HF.run() 
    mycc = cc.BCCD(myhf,conv_tol_normu=1e-8).run()
    print(mycc.e_tot)

    mo = mycc.mo_coeff
    t2 = mycc.t2

    # SPATIAL orbital numbers
    n_occ = t2.shape[0]
    n_vir = t2.shape[2]

    eri_ao = mol.intor('int2e') # two e integral
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    core_spatialorbs = mo[:, 0].reshape(-1,1)
    vir_spatialorbs = mo[:, 1:]

    # Build the required anti-symmetrised orbitals
    oovv = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spatialorbs, core_spatialorbs, anti_eri_ao, vir_spatialorbs, vir_spatialorbs, optimize="optimal")
    ovvo = np.einsum("pi,qa,pqrs,rb,sj->iabj", core_spatialorbs, vir_spatialorbs, anti_eri_ao, vir_spatialorbs, core_spatialorbs,optimize="optimal") 

    goovv = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spatialorbs, core_spatialorbs, eri_ao, vir_spatialorbs, vir_spatialorbs, optimize="optimal")
    govov = np.einsum("pi,qa,pqrs,rj,sb->iajb", core_spatialorbs, vir_spatialorbs, eri_ao, core_spatialorbs,vir_spatialorbs, optimize="optimal")

    # build self energy
    selfener_occ, selfener_vir = get_selfenergy_spatial(t2,oovv,goovv) # n_occ x n_occ, n_vir x n_vir

    # build fock matrix
    fock_occ, fock_vir = bse.build_fock_mat_bccd_spatial(mol,n_occ,n_vir) # n_occ x n_occ, n_vir x n_vir

    # build gfock
    F_ij = selfener_occ + fock_occ
    F_ab = selfener_vir + fock_vir 

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    hbse = build_bse(F_ij, F_ab, n_occ, n_vir, goovv, ovvo, govov, t2)
    
    H_bse = hbse[:,:,:,:,0].reshape((n_occ*n_vir,n_occ*n_vir))
    e, v = np.linalg.eig(H_bse)
    print(e.shape)
    print(np.real(e))

    np.savetxt("spatial_eig.csv",np.sort(np.real(e)),delimiter=',')


    