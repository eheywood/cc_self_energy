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


def build_bse_spatial(F_ij, F_ab, n_occ, n_vir, goovv, oovv, ovov, govov, t2) -> np.ndarray:

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    nspincase = 4
    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir,nspincase))
    
    for i in range(nspincase):
        H_bse[:,:,:,:,i] = F_abij - F_ijab

        if i == 0 or i == 3:
            # 0 = aaaa
            # 3 = bbbb

            # ovvo term
            H_bse[:,:,:,:,i] += ovov

            # contraction term
            term1 = np.einsum("ikbc, jkca -> iajb", oovv,t2,optimize="optimal")
            term2 = -np.einsum("ikbc, jkac -> iajb", oovv,t2,optimize="optimal")
            term3 = - np.einsum("ikbc, jkac -> iajb", goovv,t2,optimize="optimal")
            H_bse[:,:,:,:,i] += term1 + term2 + term3

        else:
            # 1 = abba
            # 2 = baab
            
            # govov term
            H_bse[:,:,:,:,i] += - govov
            # contraction term
            H_bse[:,:,:,:,i] += - np.einsum("ikcb, jkca -> iajb", goovv, t2,optimize="optimal") 

    return H_bse

def sing_excitation(hbse, n_occ_spatial, n_vir_spatial):
    hbse_new = np.zeros((n_occ_spatial,n_vir_spatial,n_occ_spatial,n_vir_spatial))
    hbse_new += hbse[:,:,:,:,0] #iajb->a,a,a,a
    hbse_new += hbse[:,:,:,:,1] #iajb->baab
    hbse_new += hbse[:,:,:,:,2] #iajb->abba
    hbse_new += hbse[:,:,:,:,3] #iajb->bbbb
    return 0.5*hbse_new.reshape(n_occ_spatial*n_vir_spatial,n_occ_spatial*n_vir_spatial)

def trip_excitation(hbse, n_occ_spatial, n_vir_spatial):
    hbse_new = np.zeros((n_occ_spatial,n_vir_spatial,n_occ_spatial,n_vir_spatial))
    hbse_new += hbse[:,:,:,:,0] #iajb->a,a,a,a
    hbse_new -= hbse[:,:,:,:,1] #iajb->baab
    hbse_new -= hbse[:,:,:,:,2] #iajb->abba
    hbse_new += hbse[:,:,:,:,3] #iajb->bbbb
    return 0.5*hbse_new.reshape(n_occ_spatial*n_vir_spatial,n_occ_spatial*n_vir_spatial)


def CC_BSE_spinfree(mol,mo,myhf,mycc,t2,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin):
    
    eri_ao = mol.intor('int2e') # two e integral
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    core_spatialorbs = mo[:, :n_occ_spatial]
    vir_spatialorbs = mo[:, n_occ_spatial:]


    # Build the required anti-symmetrised two electron integrals 
    oovv,_,_,_,ovov = helper.build_double_ints(core_spatialorbs,vir_spatialorbs,anti_eri_ao)

    # Build required two electron integrals
    goovv,_,_,_,govov = helper.build_double_ints(core_spatialorbs,vir_spatialorbs,eri_ao)

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
    F_ij_v,_,_,_,_ = dgeev(F_ij/eV2au)
    F_ab = selfener_vir + fock_vir 
    F_ab_v,_,_,_,_ = dgeev(F_ab/eV2au)

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    nspincase = 4
    hbse = build_bse_spatial(F_ij, F_ab, n_occ_spatial, n_vir_spatial, goovv, oovv, ovov, govov, t2)  # (n_occ,n_vir,n_occ,n_vir,nspincase)
    hbse_0 = hbse[:,:,:,:,1]
    hbse_eig = np.zeros(n_occ_spin*n_vir_spin)
    
    for i in range(nspincase):
        H = hbse[:,:,:,:,i].reshape(n_occ_spatial*n_vir_spatial, n_occ_spatial*n_vir_spatial)
        val = np.linalg.eigvals(H)

        hbse_eig[i*n_occ_spatial*n_vir_spatial:(i+1)*n_occ_spatial*n_vir_spatial] = np.real_if_close(val)
      
    hbse_sing = sing_excitation(hbse, n_occ_spatial, n_vir_spatial)
    hbse_trip = trip_excitation(hbse, n_occ_spatial, n_vir_spatial)

    singE, v = np.linalg.eig(hbse_sing)
    tripE, v = np.linalg.eig(hbse_trip)

#    print(f"length of single excitation:{len(singE)}")
#    print("Single excitation:")
#    print(np.real(np.sort(singE)/eV2au))
#    print("Triplet excitation:")
#    print(np.sort(tripE)/eV2au)
    

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"{label}, spin-free-orb\n")
        # f.write("Beryllium, spin-free-orb\n")
        f.write(f"Singlet exci./eV: {np.sort(np.real(singE))[:10] / eV2au}\n")
        f.write(f"Triplet exci./eV: {np.sort(np.real(tripE))[:10] / eV2au}\n")
        f.write("\n")
        
    return hbse_0, selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, F_ij_v, F_ab_v, hbse_eig, singE, tripE
