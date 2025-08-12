import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493


def build_fock_mat_bccd_spatial(mol, myhf, mycc,n_occ,n_vir,spin:bool=False)-> tuple[np.ndarray,np.ndarray]:

    # build Fock matrix
    F_ao = myhf.get_fock()    
    C  = myhf.mo_coeff        
    F_mo = C.T @ F_ao @ C   
    fock_occ = F_mo[:n_occ_spatial,:n_occ_spatial]
    fock_vir = F_mo[n_occ_spatial:,n_occ_spatial:]

    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    bmo = mycc.mo_coeff
    bmo_occ = bmo[:, :n_occ_spatial]
    bmo_vir = bmo[:, n_occ_spatial:]

    hcore_occ = np.einsum("pi,pq,qj->ij", bmo_occ, hcore, bmo_occ, optimize="optimal")
    hcore_vir = np.einsum("pi,pq,qj->ij", bmo_vir, hcore, bmo_vir, optimize="optimal")#
    eri = mol.intor("int2e").transpose(0,2,1,3)
    fock_occ = hcore_occ +\
            2 * np.einsum("pi,qk,pqrs,rj,sk->ij", bmo_occ, bmo_occ, eri, bmo_occ, bmo_occ, optimize="optimal") -\
            np.einsum("pi,qk,pqrs,sj,rk->ij", bmo_occ, bmo_occ, eri, bmo_occ, bmo_occ, optimize="optimal")
    fock_vir = hcore_vir +\
                2 * np.einsum("pi,qk,pqrs,rj,sk->ij", bmo_vir, bmo_occ, eri, bmo_vir, bmo_occ, optimize="optimal") -\
                np.einsum("pi,qk,pqrs,sj,rk->ij", bmo_vir, bmo_occ, eri, bmo_vir, bmo_occ, optimize="optimal")

    print(fock_vir.shape)
    print(fock_occ.shape)

    if spin:
        # times two for two spin cases.
        fock_occ_spin = np.zeros((n_occ*2,n_occ*2))
        fock_vir_spin = np.zeros((n_vir*2,n_vir*2))

        fock_occ_spin[:n_occ,:n_occ] = fock_occ
        fock_occ_spin[n_occ:,n_occ:] = fock_occ

        fock_vir_spin[:n_vir,:n_vir] = fock_vir
        fock_vir_spin[n_vir:,n_vir:] = fock_vir

        return fock_occ_spin, fock_vir_spin
    else:
        return fock_occ, fock_vir

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


def build_fock_matrices_spatial(mycc)-> tuple[np.ndarray,np.ndarray]:

    mf = scf.HF(mol)     
    mf.kernel()        

    myhf = mol.HF.run() 
    mycc = cc.BCCD(myhf,conv_tol_normu=1e-8).run()
      
    F_ao = mf.get_fock()    

    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    bmo = mycc.mo_coeff
    bmo_occ = bmo[:, :int(n_occ)]
    bmo_vir = bmo[:, int(n_occ):]

    hcore_occ = np.einsum("pi,pq,qj->ij", bmo_occ, hcore, bmo_occ, optimize="optimal")
    hcore_vir = np.einsum("pi,pq,qj->ij", bmo_vir, hcore, bmo_vir, optimize="optimal")#
    eri = mol.intor("int2e").transpose(0,2,1,3)
    fock_occ = hcore_occ +\
            2 * np.einsum("pi,qk,pqrs,rj,sk->ij", bmo_occ, bmo_occ, eri, bmo_occ, bmo_occ, optimize="optimal") -\
            np.einsum("pi,qk,pqrs,sj,rk->ij", bmo_occ, bmo_occ, eri, bmo_occ, bmo_occ, optimize="optimal")
    fock_vir = hcore_vir +\
                2 * np.einsum("pi,qk,pqrs,rj,sk->ij", bmo_vir, bmo_occ, eri, bmo_vir, bmo_occ, optimize="optimal") -\
                np.einsum("pi,qk,pqrs,sj,rk->ij", bmo_vir, bmo_occ, eri, bmo_vir, bmo_occ, optimize="optimal")
    return fock_occ, fock_vir



def build_bse_spatial(F_ij, F_ab, n_occ, n_vir, goovv, oovv, govov, t2) -> np.ndarray:

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
            H_bse[:,:,:,:,i] += - np.einsum("iabj->ibja", ovvo, optimize="optimal") # because we are swapping labels
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

if __name__ == "__main__":

    # Define Molecule to calculate amplitudes and mo for
    # mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
    #             basis='cc-pVTZ',
    #             spin=0,
    #             symmetry=False,
    #             unit="Bohr")


    # mol = gto.M(
    #     atom = """C -0.00234503 0.00000000 0.87125063
    #             C -1.75847785 0.00000000 -1.34973671
    #             O  2.27947397 0.00000000 0.71968028
    #             H -0.92904537 0.00000000 2.73929404
    #             H -2.97955463 1.66046488 -1.25209463
    #             H -2.97955463 -1.66046488 -1.25209463
    #             H -0.70043433 0.00000000 -3.11066412""",
    #     basis = "aug-cc-pVTZ",  
    #     spin = 0,
    #     symmetry = False,
    #     unit = "Bohr",
    # )

    # mol = gto.M(
    # atom = """N 0.12804615 0.00000000 0.00000000
    #         H -0.59303935 0.88580079 -1.53425197
    #         H -0.59303935 -1.77160157 0.00000000
    #         H -0.59303935 0.88580079 1.53425197""",
    # basis = "aug-cc-pVTZ",  
    # spin = 0,
    # symmetry = False,
    # unit = "Bohr",
    # )
    
    mol = gto.M(
    atom = """C 0.00000000 0.00000000 1.17922927
                C 0.00000000 0.00000000 -1.1792292""",
    basis = "aug-cc-pVTZ",  
    spin = 0,
    symmetry = False,
    unit = "Bohr")


    # get molecular orbitals and t2 amplitudes
    myhf = mol.HF.run() 
    mycc = cc.BCCD(myhf,max_cycle = 200, conv_tol_normu=1e-8).run()
    print(mycc.e_tot)

    mo = mycc.mo_coeff
    t2 = mycc.t2

    # SPATIAL orbital numbers
    n_occ_spatial = int(t2.shape[0])
    n_vir_spatial = int(t2.shape[2])

    eri_ao = mol.intor('int2e') # two e integral
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    core_spatialorbs = mo[:, :n_occ_spatial]
    vir_spatialorbs = mo[:, n_occ_spatial:]

    # Build the required anti-symmetrised orbitals
    oovv = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spatialorbs, core_spatialorbs, anti_eri_ao, vir_spatialorbs, vir_spatialorbs, optimize="optimal")
    ovvo = np.einsum("pi,qa,pqrs,rb,sj->iabj", core_spatialorbs, vir_spatialorbs, anti_eri_ao, vir_spatialorbs, core_spatialorbs,optimize="optimal") 

    goovv = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spatialorbs, core_spatialorbs, eri_ao, vir_spatialorbs, vir_spatialorbs, optimize="optimal")
    govov = np.einsum("pi,qa,pqrs,rj,sb->iajb", core_spatialorbs, vir_spatialorbs, eri_ao, core_spatialorbs,vir_spatialorbs, optimize="optimal")

    # build self energy
    selfener_occ, selfener_vir = get_selfenergy_spatial(t2,oovv,goovv) # n_occ x n_occ, n_vir x n_vir

    # build fock matrix
    fock_occ, fock_vir = build_fock_mat_bccd_spatial(mol, myhf, mycc,n_occ_spatial,n_vir_spatial,spin=False) # n_occ x n_occ, n_vir x n_vir

    # build gfock
    F_ij = selfener_occ + fock_occ
    F_ab = selfener_vir + fock_vir 

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    hbse = build_bse_spatial(F_ij, F_ab, n_occ_spatial, n_vir_spatial, goovv, oovv, govov, t2)
    np.savetxt("spatial_eig.csv", np.linalg.eig(hbse[:,:,:,:,0].reshape(n_occ_spatial*n_vir_spatial, n_occ_spatial*n_vir_spatial)))

    hbse_sing = sing_excitation(hbse, n_occ_spatial, n_vir_spatial)
    hbse_trip = trip_excitation(hbse, n_occ_spatial, n_vir_spatial)

    singE, v = np.linalg.eig(hbse_sing)
    tripE, v = np.linalg.eig(hbse_trip)

    print(f"length of single excitation:{len(singE)}")
    print("Single excitation:")
    print(np.real(np.sort(singE)/eV_to_Hartree))
    print("Triplet excitation:")
    print(np.sort(tripE)/eV_to_Hartree)


    with open("results.txt", "a", encoding="utf-8") as f:
        f.write("acetaldehyde, spin-free-orb\n")
        f.write(f"Singlet exci./eV: {np.sort(np.real(singE))[:10] / eV_to_Hartree}\n")
        f.write(f"Triplet exci./eV: {np.sort(np.real(tripE))[:10] / eV_to_Hartree}\n")
        f.write("\n")
