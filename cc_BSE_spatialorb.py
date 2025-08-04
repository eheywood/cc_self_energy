import numpy as np
from pyscf import gto, scf, cc
from scipy.linalg import block_diag
from BSE_Helper import spinor_one_and_two_e_int

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def get_selfenergy_spatial(t2,oovv, goovv):
    selfener_occ_1 = np.einsum('ikab, ijab -> ij', oovv, t2, optimize="optimal")
    selfener_occ_2 = -np.einsum('ikab, ijba -> ij', oovv, t2, optimize="optimal")
    selfener_occ_3 = np.einsum('ikab, jkab -> ij', goovv, t2, optimize="optimal")
    selfener_occ_4 = np.einsum('ikba, jkba -> ij', goovv, t2, optimize="optimal")

    selfener_occ = 0.5*(selfener_occ_1 + selfener_occ_2 + selfener_occ_3 + selfener_occ_4)
    
    selfener_vir_1 = np.einsum('ijbc, ijac -> ab', oovv, t2, optimize="optimal")
    selfener_vir_2 = - np.einsum('ijbc, ijca -> ab', oovv, t2, optimize="optimal")
    selfener_vir_3 = np.einsum('ijbc, ijab -> ab', goovv, t2, optimize="optimal")
    selfener_vir_4 = np.einsum('ijcb, ijca -> ab', goovv, t2, optimize="optimal")

    selfener_vir = -0.5*(selfener_vir_1 + selfener_vir_2 + selfener_vir_3 + selfener_vir_4)
    return selfener_occ, selfener_vir


def build_fock_matrices_spatial(mol)-> tuple[np.ndarray,np.ndarray]:

    mf = scf.HF(mol)     
    mf.kernel()          
    F_ao = mf.get_fock()    
    C   = mf.mo_coeff        
    F_mo = C.T @ F_ao @ C   
    fock_occ = F_mo[:int(n_occ),:int(n_occ)]
    fock_vir = F_mo[int(n_occ):,int(n_occ):]

    return fock_occ, fock_vir

def build_gfock_spatial(t2, oovv, goovv, selfener_occ, selfener_vir, fock_occ, fock_vir) -> tuple[np.ndarray,np.ndarray]:
    # n_occ x n_occ, n_vir x n_vir
    occ_selfeng, vir_selfeng = get_selfenergy_spatial(t2,oovv, goovv)

    # n_occ x n_occ, n_vir x n_vir
    fock_occ, fock_vir = build_fock_matrices_spatial(mol) 
    
    F_ij = occ_selfeng + fock_occ
    F_ab = vir_selfeng + fock_vir 

    return F_ij, F_ab

def build_bse(F_ij, F_ab, n_occ, n_vir) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    nspincase = 2
    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir,nspincase))
    
    for i in range(2):
        H_bse[:,:,:,:,i] = F_abij - F_ijab + np.einsum("iabj->iajb", ovvo,optimize='optimal')

        for ispina in range(2):
            for ispini in range (2):
                if ispina == ispini:
                    # ispincase = 1  or 2
                    H_bse[:,:,:,:,i] += - np.einsum("ikcb, jkca -> iajb", goovv,t2,optimize="optimal")
    
                else:
                    # ispincase = 3 or 4
                    H_bse[:,:,:,:,i] += - np.einsum("ikbc, jkac -> iajb", goovv,t2,optimize="optimal") + np.einsum("ikbc, jkca -> iajb", oovv,t2,optimize="optimal")
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
    mo = mycc.mo_coeff
    t2 = mycc.t2

    # spatial orbital numbers
    n_occ = t2.shape[0]
    n_vir = t2.shape[2]

    eri_ao = mol.intor('int2e') # two e integral
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    core_spatialorbs = mo[:, 0].reshape(-1,1)
    vir_spatialorbs = mo[:, 1:]

    # Build the required anti-symmetrised orbitals
    oovv = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spatialorbs, core_spatialorbs, anti_eri_ao, vir_spatialorbs, vir_spatialorbs, optimize="optimal")
    goovv = np.einsum("pi,qa,pqrs,rj,sb->iajb", core_spatialorbs, core_spatialorbs, eri_ao, vir_spatialorbs, vir_spatialorbs, optimize="optimal")

    # build self energy
    selfener_occ, selfener_vir = get_selfenergy_spatial(t2,oovv, goovv) # n_occ x n_occ, n_vir x n_vir

    # build fock matrix
    fock_occ, fock_vir = build_fock_matrices_spatial(mol) # n_occ x n_occ, n_vir x n_vir

    # build gfock
    F_ij = selfener_occ + fock_occ
    F_ab = selfener_vir + fock_vir 

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    # hbse = build_bse(occ_selfeng, vir_selfeng, gfock_occ, gfock_vir, n_occ, n_vir)
    
    #H_bse = H_bse.reshape((n_occ*n_vir,n_occ*n_vir))
    #e, v = np.linalg.eig(hbse)


    


