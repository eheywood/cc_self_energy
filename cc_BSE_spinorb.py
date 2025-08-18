import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as helper
from scipy.linalg.lapack import dgeev 


def get_self_energy(t2_spin:np.ndarray, oovv:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Calculates the self energy using t2 CC amplitudes. Eq 49 and 50 in the Paper.

    Parameters
    ----------
    t2 : np.ndarray
        t2 amplitudes
    oovv : np.ndarray
        anti symmetrised integral of occupied, occupied, virtual, virtual

    Returns
    -------
    tuple[np.ndarray,np.ndarray]
        occupied self energy, virtual self energy 
    """

    occ_selfeng = 0.5 * np.einsum("ikab,jkab -> ij", oovv,t2_spin,optimize="optimal")
    vir_selfeng = -0.5 * np.einsum("ijbc,ijac -> ab", oovv,t2_spin,optimize="optimal")

    return occ_selfeng, vir_selfeng


def build_gfock(mol,myhf,mycc,t2_spin, oovv, n_occ_spatial,n_vir_spatial) -> tuple[np.ndarray,np.ndarray]:
    occ_selfeng, vir_selfeng = get_self_energy(t2_spin,oovv) # n_occ x n_occ, n_vir x n_vir

    fock_occ, fock_vir = helper.bccd_fock_mat(mol,myhf,mycc,n_occ_spatial,n_vir_spatial,spin=True) 

    F_ij = occ_selfeng + fock_occ    #(n_occ x n_occ)
    F_ab = vir_selfeng + fock_vir    #(n_vir x n_vir)

    F_ij_v,_,_,_,_ = dgeev(F_ij)
    diagFij = np.diag(F_ij_v)    
    F_ab_v,_,_,_,_ = dgeev(F_ab)
    diagFab = np.diag(F_ab_v)

    return diagFij,diagFab

def build_bse(F_ij:np.ndarray,F_ab,ovvo,oovv,t2, n_occ, n_vir) -> np.ndarray:
    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir))
    term1 = - np.einsum('iabj->iajb', ovvo, optimize='optimal') # because we are swapping the sign
    H_bse = term1 
    #+ np.einsum("ikbc, jkca ->iajb",oovv, t2, optimize='optimal')

    #H_bse = F_abij - F_ijab + term1 + np.einsum("ikbc, jkca ->iajb",oovv, t2, optimize='optimal')
    
    return H_bse

def CC_BSE_spin(mol,mo,myhf,mycc,label,eV2au,n_occ_spatial,n_vir_spatial, n_occ_spin, n_vir_spin):
    
    # get molecular orbitals and t2 amplitudes
    _, t2, _, _, _, _ = helper.bccd_t2_amps(mycc,myhf)
 
    core_spinorbs, vir_spinorbs = helper.get_spinorbs(mo,n_occ_spatial)
    #n_occ = core_spinorbs.shape[1]
    #n_vir = vir_spinorbs.shape[1]

    # Build electron repulsion integrals
    _, eri_ao = helper.spinor_one_and_two_e_int(mol)                # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # Build the required anti-symmetrised orbitals
    oovv,_,_,ovvo,ovov = helper.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    
    #debugging ###########################################
    def spin_output(t2,mol,myhf,mycc,oovv,n_occ_spatial,n_vir_spatial):
      selfener_occ_spin, selfener_vir_spin = get_self_energy(t2, oovv)
      fock_occ_spin, fock_vir_spin = helper.bccd_fock_mat(mol,myhf,mycc,n_occ_spatial,n_vir_spatial,spin=True)
      return selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin
    selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin  = spin_output(t2,mol,myhf,mycc,oovv,n_occ_spatial,n_vir_spatial)
    ###########################################

    # n_occ x n_occ, n_vir x n_vir 
    # Extended fock operator (fock + self energy)
    gfock_occ, gfock_vir = build_gfock(mol,myhf,mycc,t2,oovv,n_occ_spatial,n_vir_spatial)

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    hbse = build_bse(gfock_occ,gfock_vir,ovvo,oovv,t2,n_occ_spin,n_vir_spin)
    H = hbse.reshape((n_occ_spin*n_vir_spin,n_occ_spin*n_vir_spin))
    
    val = np.linalg.eigvals(H)

    hbse_sing = helper.sing_excitation(hbse, n_occ_spatial, n_vir_spatial)
    hbse_trip = helper.trip_excitation(hbse, n_occ_spatial, n_vir_spatial)
    #hbse_trip2 = trip_excitation_spin(hbse, n_occ_spatial, n_vir_spatial)
    singE, _ = np.linalg.eig(hbse_sing)
    tripE, _ = np.linalg.eig(hbse_trip)
    #tripE2, _ = np.linalg.eig(hbse_trip2)
    #print(f"length of single excitation:{len(singE)}")
    #print("Singlet excitation:")
    #print(np.sort(singE)/eV2au)
    #print("Triplet excitation:")
    #print(np.sort(tripE)/eV2au)

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"{label}, spin-orb\n")
        #f.write("Beryllium, spin-orb\n")
        f.write(f"Singlet exci./eV: {np.sort(np.real(singE))[:10] / eV2au}\n")
        f.write(f"Triplet exci./eV: {np.sort(np.real(tripE))[:10] / eV2au}\n")
        f.write("\n")
        
    return selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, gfock_occ, gfock_vir, val, singE, tripE