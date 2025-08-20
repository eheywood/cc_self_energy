import numpy as np
import src.BSE_Helper as helper
from scipy.linalg.lapack import dgeev 


def get_selfenergy_spatial(t2,oovv, goovv):
    selfener_occ_1 = np.einsum('ikab, jkab -> ij', oovv, t2, optimize="optimal")
    selfener_occ_2 = -np.einsum('ikab, jkba -> ij', oovv, t2, optimize="optimal")
    selfener_occ_3 = np.einsum('ikab, jkab -> ij', goovv, t2, optimize="optimal")
    selfener_occ_4 = np.einsum('ikba, jkab -> ij', goovv, t2, optimize="optimal")
    selfener_occ = 0.5*(selfener_occ_1 + selfener_occ_2 + selfener_occ_3 + selfener_occ_4)


    selfener_vir_1 = np.einsum('ijbc, ijac -> ab', oovv, t2, optimize="optimal")
    selfener_vir_2 = - np.einsum('ijbc, ijca -> ab', oovv, t2, optimize="optimal")
    selfener_vir_3 = np.einsum('ijbc, ijac -> ab', goovv, t2, optimize="optimal")
    selfener_vir_4 = np.einsum('ijcb, ijca -> ab', goovv, t2, optimize="optimal")
    selfener_vir = -0.5*(selfener_vir_1 + selfener_vir_2 + selfener_vir_3 + selfener_vir_4)
    return selfener_occ, selfener_vir


def build_bse_spatial(F_ij, F_ab, n_occ, n_vir, goovv, oovv, ovvo, govvo, t2):

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    nspincase = 4
    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir,nspincase))
    term1_sum = np.zeros((n_occ,n_vir,n_occ,n_vir,nspincase))
    term2_sum = np.zeros((n_occ,n_vir,n_occ,n_vir,nspincase))
    
    for i in range(nspincase):
        H_bse[:,:,:,:,i] = F_abij - F_ijab
        term1 = np.zeros((n_occ,n_vir,n_occ,n_vir))
        term2 = np.zeros((n_occ,n_vir,n_occ,n_vir))
        if i == 0 or i == 3:
            # 0 = aaaa
            # 3 = bbbb

            # ovvo term
            term1 =  - np.einsum('iabj->iajb', ovvo, optimize='optimal')

            # contraction term
            term2 = np.einsum("ikbc, jkac -> iajb", oovv,t2,optimize="optimal") \
                   -np.einsum("ikbc, jkca -> iajb", oovv,t2,optimize="optimal") \
                   - np.einsum("ikbc, jkac -> iajb", goovv,t2,optimize="optimal")


        elif i == 1:
            # 1 = abba
            # 2 = baab
            
            # govov term
            term1 = - np.einsum('iabj->iajb', ovvo, optimize='optimal')
            #term1 += np.einsum('iabj->iajb', govvo, optimize='optimal')
            # contraction term
            term2 = - np.einsum("ikcb, jkca -> iajb", goovv, t2,optimize="optimal") 
        

        H_bse[:,:,:,:,i] += term1 + term2 
        term1_sum[:,:,:,:,i] += term1
        term2_sum[:,:,:,:,i] += term2

    return term1_sum, term2_sum, H_bse

def build_hbse_singtrip(F_ij, F_ab, nc, nv, t2, govvo, govov, goovv):

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(nc),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(nv),optimize='optimal')

    hbseSing = np.zeros((nc,nv,nc,nv))
    hbseTrip = np.zeros((nc,nv,nc,nv))
    
    fock = F_abij - F_ijab

    iabj = 2*np.einsum('iabj->iajb', govvo, optimize='optimal')
    iajb = - govov

    ikbc_ikcb = 2*goovv - np.einsum("ikcb->ikbc", goovv, optimize='optimal')
    term3 = np.einsum("ikbc,jkca->iajb", ikbc_ikcb, t2, optimize='optimal')

    hbseSing = fock + iabj + iajb + term3
    hbseTrip = fock + iajb - np.einsum("ikcb,jkca->iajb", goovv, t2, optimize='optimal')

    SingEner, v = np.linalg.eig(hbseSing.reshape((nc*nv, nc*nv)))
    TripEner, v = np.linalg.eig(hbseTrip.reshape((nc*nv, nc*nv)))

    return np.sort(np.real_if_close(SingEner)), np.sort(np.real_if_close(TripEner)) # in eV



def sing_excitation(hbse, n_occ, n_vir):
    """Build singlet excitation Hamiltonian."""
    hbse_new = hbse[..., 0] + hbse[..., 1] + hbse[..., 2] + hbse[..., 3]
    return 0.5 * hbse_new.reshape(n_occ * n_vir, n_occ * n_vir)


def trip_excitation(hbse, n_occ, n_vir):
    """Build triplet excitation Hamiltonian."""
    hbse_new = hbse[..., 0] - hbse[..., 1] - hbse[..., 2] + hbse[..., 3]
    return 0.5 * hbse_new.reshape(n_occ * n_vir, n_occ * n_vir)


def CC_BSE_spinfree(mol,mo,myhf,mycc,t2,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin):
    
    eri_ao = mol.intor('int2e') # two e integral
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    core_spatialorbs = mo[:, :n_occ_spatial]
    vir_spatialorbs = mo[:, n_occ_spatial:]


    # Build the required anti-symmetrised two electron integrals 
    oovv,_,_,ovvo,ovov = helper.build_double_ints(core_spatialorbs,vir_spatialorbs,anti_eri_ao)

    # Build required two electron integrals
    goovv,_,_,govvo,govov = helper.build_double_ints(core_spatialorbs,vir_spatialorbs,eri_ao)

    # build self energy
    selfener_occ, selfener_vir = get_selfenergy_spatial(t2,oovv,goovv) # n_occ x n_occ, n_vir x n_vir

    # build fock matrix
    fock_occ, fock_vir = helper.bccd_fock_mat(mol, myhf, mycc,n_occ_spatial, n_vir_spatial,spin=False) # n_occ x n_occ, n_vir x n_vir
    
    
    #debugging###########################################
    def spa_output(mol, myhf, mycc,n_occ_spatial,n_vir_spatial,t2, oovv, goovv):
      selfener_occ_spa, selfener_vir_spa = get_selfenergy_spatial(t2,oovv, goovv)
      fock_occ_spa, fock_vir_spa = helper.bccd_fock_mat(mol, myhf, mycc,n_occ_spatial,n_vir_spatial,spin=False)
      return selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa
    selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa  = spa_output(mol, myhf, mycc,n_occ_spatial,n_vir_spatial,t2, oovv, goovv)
    #####################################################

    # build gfock
    F_ij = selfener_occ + fock_occ
    F_ab = selfener_vir + fock_vir 

    F_ij_v,_,_,_,_ = dgeev(F_ij)
    gfock_occ = np.sort(F_ij_v) 
    F_ab_v,_,_,_,_ = dgeev(F_ab) 
    gfock_vir = np.sort(F_ab_v)

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    
    term1_sum, term2_sum, hbse = build_bse_spatial(F_ij, F_ab, n_occ_spatial, n_vir_spatial, goovv, oovv, ovvo, govvo, t2)  # (n_occ,n_vir,n_occ,n_vir,nspincase)

    SingEner, TripEner = build_hbse_singtrip(F_ij, F_ab, n_occ_spatial, n_vir_spatial, t2, govvo, govov, goovv)

    eig, term1_diag, term2_diag = [], [], []
    for i in range(4): 
        H = hbse[:,:,:,:,i].reshape(n_occ_spatial*n_vir_spatial, n_occ_spatial*n_vir_spatial) 
        val = np.linalg.eigvals(H) 
        eig += list(np.real_if_close(val)) 
        H_term1 = term1_sum[:,:,:,:,i].reshape(n_occ_spatial*n_vir_spatial, n_occ_spatial*n_vir_spatial) 
        val1 = np.linalg.eigvals(H_term1) 
        term1_diag += list(np.real_if_close(val1)) 
        H_term2 = term2_sum[:,:,:,:,i].reshape(n_occ_spatial*n_vir_spatial, n_occ_spatial*n_vir_spatial) 
        val2 = np.linalg.eigvals(H_term2) 
        term2_diag += list(np.real_if_close(val2))

    # # Singlet/Triplet Excitations
    # singE, _ = np.linalg.eig(sing_excitation(term1_sum, n_occ_spatial, n_vir_spatial))
    # tripE, _ = np.linalg.eig(trip_excitation(term1_sum, n_occ_spatial, n_vir_spatial))

    # with open("results.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{label}, spin-free-orb\n")
    #     # f.write("Beryllium, spin-free-orb\n")
    #     f.write(f"Singlet exci./eV: {np.sort(np.real(singE))[:10] / eV2au}\n")
    #     f.write(f"Triplet exci./eV: {np.sort(np.real(tripE))[:10] / eV2au}\n")
    #     f.write("\n")
        
    return selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, gfock_occ, gfock_vir, eig, SingEner, TripEner
