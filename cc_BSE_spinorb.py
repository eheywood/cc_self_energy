import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def build_gfock(t2, oovv, n_occ,n_vir) -> tuple[np.ndarray,np.ndarray]:
    # n_occ x n_occ, n_vir x n_vir
    occ_selfeng, vir_selfeng = bse.get_self_energy(t2,oovv)

    # n_occ x n_occ, n_vir x n_vir
    spat_occ = int(n_occ/2)
    spat_vir = int(n_vir/2)
    fock_occ, fock_vir = bse.build_fock_mat_bccd_spatial(mol,spat_occ,spat_vir,spin=True) 

    F_ij = occ_selfeng + fock_occ
    F_ab = vir_selfeng + fock_vir

    return F_ij,F_ab

def build_bse(F_ij:np.ndarray,F_ab,ovvo,oovv,t2, n_occ, n_vir) -> np.ndarray:
    
    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')

    H_bse = np.zeros((n_occ,n_vir,n_occ,n_vir))
    
    H_bse = F_abij - F_ijab + np.einsum("iabj->iajb", ovvo,optimize='optimal') + np.einsum("ikbc, jkca ->ibja",oovv, t2, optimize='optimal')
    
    return H_bse

def sing_excitation(hbse, n_occ_spatial, n_vir_spatial):
    hbse_new = np.zeros((n_occ_spatial,n_vir_spatial,n_occ_spatial,n_vir_spatial))
    hbse_new += hbse[:n_occ_spatial,:n_vir_spatial,:n_occ_spatial,:n_vir_spatial] #iajb->a,a,a,a
    hbse_new += hbse[n_occ_spatial:,:n_vir_spatial,:n_occ_spatial,n_vir_spatial:] #iajb->baab
    hbse_new += hbse[:n_occ_spatial,n_vir_spatial:,n_occ_spatial:,:n_vir_spatial] #iajb->abba
    hbse_new += hbse[n_occ_spatial:,n_vir_spatial:,n_occ_spatial:,n_vir_spatial:] #iajb->bbbb
    
    return 0.5*hbse_new.reshape(n_occ_spatial*n_vir_spatial,n_occ_spatial*n_vir_spatial)

def trip_excitation(hbse, n_occ_spatial, n_vir_spatial):
    hbse_new = np.zeros((n_occ_spatial,n_vir_spatial,n_occ_spatial,n_vir_spatial))
    hbse_new += hbse[:n_occ_spatial,:n_vir_spatial,:n_occ_spatial,:n_vir_spatial] #iajb->a,a,a,a
    hbse_new -= hbse[n_occ_spatial:,:n_vir_spatial,:n_occ_spatial,n_vir_spatial:] #iajb->baab
    hbse_new -= hbse[:n_occ_spatial,n_vir_spatial:,n_occ_spatial:,:n_vir_spatial] #iajb->abba
    hbse_new += hbse[n_occ_spatial:,n_vir_spatial:,n_occ_spatial:,n_vir_spatial:] #iajb->bbbb
    
    return 0.5*hbse_new.reshape(n_occ_spatial*n_vir_spatial,n_occ_spatial*n_vir_spatial)

def trip_excitation2(hbse, n_occ_spatial, n_vir_spatial):
    hbse_new = np.zeros((n_occ_spatial,n_vir_spatial,n_occ_spatial,n_vir_spatial))
    hbse_new += hbse[:n_occ_spatial,n_vir_spatial:,:n_occ_spatial,n_vir_spatial:] #iajb->abab
    hbse_new -= hbse[:n_occ_spatial,n_vir_spatial:,n_occ_spatial:,:n_vir_spatial] #iajb->abba
    hbse_new -= hbse[n_occ_spatial:,:n_vir_spatial,:n_occ_spatial,n_vir_spatial:] #iajb->baab
    hbse_new += hbse[n_occ_spatial:,:n_vir_spatial,n_occ_spatial:,:n_vir_spatial] #iajb->baba
    
    return 0.5*hbse_new.reshape(n_occ_spatial*n_vir_spatial,n_occ_spatial*n_vir_spatial)


if __name__ == "__main__":

    # #Define Molecule to calculate amplitudes and mo for
    # mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
    #             basis='cc-pVTZ',
    #             spin=0,
    #             symmetry=False,
    #             unit="Bohr")

    mol = gto.M(atom="C -0.00234503 0.00000000 0.87125063 \
                C -1.75847785 0.00000000 -1.34973671 \
                O 2.27947397 0.00000000 0.71968028 \
                H -0.92904537 0.00000000 2.73929404 \
                H -2.97955463 1.66046488 -1.25209463 \
                H -2.97955463 -1.66046488 -1.25209463 \
                H -0.70043433 0.00000000 -3.11066412",
            basis='aug-cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr")

    # get molecular orbitals and t2 amplitudes
    mo,t2,_,_,n_occ_spatial,n_vir_spatial = bse.bccd_t2_amps(mol)

    print(t2.shape)  
    core_spinorbs, vir_spinorbs = bse.get_spinorbs(mo, t2.shape, n_occ_spatial,n_vir_spatial)
    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]
    n_occ_spatial = int(core_spinorbs.shape[1]/2)
    n_vir_spatial = int(vir_spinorbs.shape[1]/2)

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
    # hbse = hbse.reshape((n_occ*n_vir,n_occ*n_vir))
    
    #e, v = np.linalg.eig(hbse)

    hbse_sing = sing_excitation(hbse, n_occ_spatial, n_vir_spatial)
    hbse_trip = trip_excitation(hbse, n_occ_spatial, n_vir_spatial)
    #hbse_trip2 = trip_excitation(hbse, n_occ_spatial, n_vir_spatial)
    singE, _ = np.linalg.eig(hbse_sing)
    tripE, _ = np.linalg.eig(hbse_trip)
    #tripE2, _ = np.linalg.eig(hbse_trip2)
    print(f"length of single excitation:{len(singE)}")
    print("Singlet excitation:")
    print(np.sort(singE)/eV_to_Hartree)
    print("Triplet excitation:")
    print(np.sort(tripE)/eV_to_Hartree)

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write("acetaldehyde, spin-orb\n")
        f.write(f"Singlet exci./eV: {np.sort(np.real(singE))[:10] / eV_to_Hartree}\n")
        f.write(f"Triplet exci./eV: {np.sort(np.real(tripE))[:10] / eV_to_Hartree}\n")
        f.write("\n")