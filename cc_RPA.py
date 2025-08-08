import numpy as np
from pyscf import gto
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def get_GW_BSE_amps(X_rpa,Y_rpa,eig_rpa,vir_gwe, core_gwe,ooov_anti,vovv_anti,oovv_anti) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    
    # Build W

    #Build the transfer coefficients
    M_ijm = np.einsum("ikjc,kcm->ijm",ooov_anti,X_rpa+Y_rpa,optimize='optimal')
    M_abm = np.einsum("akbc,kcm->abm",vovv_anti,X_rpa+Y_rpa,optimize='optimal') 
    M_iam = np.einsum("ikac,kcm->iam",oovv_anti,X_rpa+Y_rpa,optimize='optimal') 
    M_jbm = M_iam

    inv_eig = 1/eig_rpa
    W_iajb_correction = -2*(np.einsum("ijm,abm,m -> iajb",M_ijm,M_abm,inv_eig,optimize='optimal'))
    W_ijba_correction = -2*(np.einsum("iam,jbm,m -> ijba",M_iam,M_jbm,inv_eig,optimize='optimal'))

    # Using W, find new X and Y
    gwe_diff = vir_gwe.reshape(-1,1) - core_gwe
    AW = -np.einsum("iabj->iajb",ovvo_anti,optimize='optimal') - W_iajb_correction
    AW += np.einsum("ai,ab,ij-> iajb", gwe_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')
    BW = -np.einsum("ijab->iajb",oovv_anti,optimize='optimal') - np.einsum("ijba->iajb",W_ijba_correction,optimize='optimal')

    eigW, XW, YW = bse.super_matrix_solver(AW,BW)
    print("GWE-BSE Complete")

    return eigW, XW, YW

def get_RPA_amps(vir_e, core_e, ovvo_anti, oovv_anti, n_occ,n_vir) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # construct A and B and use supermatrix solver to get eigenvectors and values
    e_diff = vir_e.reshape(-1,1) - core_e
    A = -np.einsum("iabj->iajb", ovvo_anti) 
    A += np.einsum("ai,ab,ij-> iajb", e_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')
    B = -np.einsum("ijab->iajb", oovv_anti)

    print(A.shape)
    print(B.shape)

    eig, X, Y = bse.super_matrix_solver(A,B)
    print('RPA COMPLETE')
    return eig, X, Y, A, B


def build_RPA_hamiltonian(vir_e, core_e, ovvo_anti, oovv_anti,n_occ,n_vir,t2) -> np.ndarray:
    
    H_rpa = np.zeros((n_occ,n_vir,n_occ,n_vir))
    e_diff = vir_e.reshape(-1,1) - core_e

    term_1 = np.einsum("ai,ab,ij->iajb", e_diff,np.identity(n_vir),np.identity(n_occ),optimize='optimal')
    term_2 = -np.einsum("iabj->iajb", ovvo_anti,optimize='optimal')
    term_3 = np.einsum("ikbc,jkca->iajb",oovv_anti,t2,optimize='optimal')
    H_rpa = term_1 + term_2 + term_3

    return H_rpa

if __name__ == "__main__":

    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
            basis='cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr")
    
    # BCCD MO's and amplitudes. (use in GW BSE)
    # mo,t2,core_e,vir_e = bse.bccd_t2_amps(mol)

    # HF molecular orbitals (use in RPA only)
    myhf = mol.HF.run() 
    mo = myhf.mo_coeff

    # Orbital energies
    core_e = np.array(list(myhf.mo_energy[:1]) + list(myhf.mo_energy[:1]))
    vir_e = np.array(list(myhf.mo_energy[1:]) + list(myhf.mo_energy[1:]))

    core_spinorbs, vir_spinorbs =  bse.get_spinorbs(mo)

    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    # Constructing <ij|ab> and <ia|bj>
    _, eri_ao = bse.spinor_one_and_two_e_int(mol)                   # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # Build the required anti-symmetrised orbitals
    oovv_anti,ooov_anti,vovv_anti,ovvo_anti,ovov_anti = bse.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # Self energies (in eV) (for GW-BSE)
    # core_gwe, vir_gwe = bse.get_self_energy(t2,oovv_anti)
    # core_gwe = np.diag(core_gwe)
    # vir_gwe = np.diag(vir_gwe)

    # Solve RPA equations to get amplitudes
    rpa_eig, X_rpa, Y_rpa, A, B = get_RPA_amps(vir_e,core_e,ovvo_anti,oovv_anti,n_occ,n_vir)

    nA = int(np.sqrt(A.size))
    nB = int(np.sqrt(B.size))
    A = A.reshape((nA, nA))
    B = B.reshape((nB, nB))

    t_coeffic = Y_rpa@np.linalg.inv(X_rpa)
    # t_coeffic = t_coeffic.reshape((n_occ,n_occ,n_vir,n_vir))

    # H_rpa = build_RPA_hamiltonian(vir_e,core_e,ovvo_anti,oovv_anti,n_occ,n_vir, t_coeffic)
    H_rpa = A + B@t_coeffic
    # H_rpa = H_rpa.reshape((n_occ*n_vir, n_occ*n_vir))
    e, _ = np.linalg.eig(H_rpa)

    print("Average diff: ")
    print(np.average(np.absolute(np.real(np.sort(rpa_eig)-np.sort(e)))))





