import numpy as np
import BSE_Helper as helper

# def get_Wijba(oovv, X, Y, rpa_ev):
#     inv_e = 1 / rpa_ev
#     M_ia = np.einsum("ikbc,kcv->ibv", oovv, X, optimize="optimal") +\
#            np.einsum("ikbc,kcv->ibv", oovv, Y, optimize="optimal")
#     return - 2 * np.einsum("ibv,jav,v->ijba", M_ia, M_ia, inv_e, optimize="optimal")

# def get_Wiajb(ooov, vovv, X, Y, rpa_ev):
#     inv_e = 1 / rpa_ev
#     M_ij = np.einsum("ikjc,kcv->ijv", ooov, X, optimize="optimal") +\
#            np.einsum("ikjc,kcv->ijv", ooov, Y, optimize="optimal")
#     M_ab = np.einsum("akbc,kcv->abv", vovv, X, optimize="optimal") +\
#            np.einsum("akbc,kcv->abv", vovv, Y, optimize="optimal")
#     return - 2 * np.einsum("ijv,abv,v->iajb", M_ij, M_ab, inv_e, optimize="optimal")


def get_GW_BSE_amps(X_rpa,Y_rpa,eig_rpa,ooov_anti,vovv_anti,oovv_anti,ovvo_anti,n_occ,n_vir,t2) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    # USE RPA eigenvectors to get the GWE amplitudes

    core_gwe, vir_gwe = helper.get_self_energy(t2,oovv_anti)
    
    m_len = eig_rpa.shape[0]
    X_rpa = X_rpa.reshape((n_occ,n_vir,m_len))
    Y_rpa = Y_rpa.reshape((n_occ,n_vir,m_len))
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

    eigW, XW, YW = helper.super_matrix_solver(AW,BW)
    print("GWE-BSE Complete")

    return eigW, XW, YW

def GW_BSE(mol,myhf,X_rpa,Y_rpa,eig_rpa,n_occ_spatial) -> tuple[np.ndarray,np.ndarray,np.ndarray]:

    mo = myhf.mo_coeff
    core_spinorbs, vir_spinorbs = helper.get_spinorbs(mo, n_occ_spatial)

    n_mo = myhf.mo_coeff.shape[1]
    n_occ_spatial = mol.nelec[0]  # Number of occupied orbitals (alpha electrons in RHF)
    n_vir_spatial = n_mo - n_occ_spatial  # Number of unoccupied orbitals

    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    # Orbital energies

    # Constructing <ij|ab> and <ia|bj>
    _, eri_ao = helper.spinor_one_and_two_e_int(mol)                   # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # Build the required anti-symmetrised orbitals
    oovv,ooov,vovv,ovvo,ovov = helper.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # RPA Amps:
    rpa_amps = Y_rpa@np.linalg.inv(X_rpa)
    rpa_amps = rpa_amps.reshape((n_occ,n_vir,n_occ,n_vir))
    rpa_amps = np.einsum("iajb->ijba",rpa_amps,optimize='optimal')
    
    eigW, XW, YW = get_GW_BSE_amps(X_rpa,Y_rpa,eig_rpa,ooov,vovv,oovv,ovvo,n_occ,n_vir,rpa_amps)

    return eigW, XW, YW
