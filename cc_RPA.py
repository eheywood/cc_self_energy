import numpy as np
from pyscf import gto
from scipy.linalg import block_diag
import BSE_Helper as bse

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

if __name__ == "__main__":

    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
            basis='cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr")
    
    mo,t2,core_e,vir_e = bse.bccd_t2_amps(mol)

    core_spinorbs, vir_spinorbs =  bse.get_spinorbs(mo)

    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    # Constructing <ij|ab> and <ia|bj>
    _, eri_ao = bse.spinor_one_and_two_e_int(mol)                   # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # <ij|ab>
    ijab = np.einsum("pi,qj,pqrs,ra,sb->ijab",core_spinorbs,core_spinorbs,eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal")
    #<ia|bj>
    iabj =  np.einsum("pi,qa,pqrs,rb,sj->iabj",core_spinorbs,vir_spinorbs,eri_ao,vir_spinorbs,core_spinorbs,optimize="optimal")
    #<ia|bj>
    iajb =  np.einsum("pi,qa,pqrs,rj,sb->iajb",core_spinorbs,vir_spinorbs,eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal")

    # Build the required anti-symmetrised orbitals
    oovv_anti,ooov_anti,vovv_anti,ovvo_anti,ovov_anti = bse.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # Self energies (in eV)
    core_gwe, vir_gwe = bse.get_self_energy(t2,oovv_anti)
    core_gwe = np.diag(core_gwe)
    vir_gwe = np.diag(vir_gwe)
    print(core_gwe)
    print(vir_gwe)

    # Solve RPA equation to get W
    # construct A and B and use supermatrix solver to get eigenvectors and values
    e_diff = vir_e.reshape(-1,1) - core_e
    A = -np.einsum("iabj->iajb", ovvo_anti) 
    A += np.einsum("ai,ab,ij-> iajb", e_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')
    B = -np.einsum("ijab->iajb", oovv_anti)

    eig, X, Y = bse.super_matrix_solver(A,B)

    m_len = eig.shape[0]
    X = X.reshape((n_occ,n_vir,m_len))
    Y = Y.reshape((n_occ,n_vir,m_len))
    print('RPA COMPLETE')

    # Build W
    #Build the transfer coefficients
    M_ijm = np.einsum("ikjc,kcm->ijm",ooov_anti,X+Y,optimize='optimal')
    # print("M_ijm: ", max(np.abs(X.reshape(-1))))
    M_abm = np.einsum("akbc,kcm->abm",vovv_anti,X+Y,optimize='optimal') 
    # print("M_abm: ", max(np.abs(Y.reshape(-1))))
    M_iam = np.einsum("ikac,kcm->iam",oovv_anti,X+Y,optimize='optimal') 
    M_jbm = M_iam

    inv_eig = 1/eig
    W_iajb_correction = -2*(np.einsum("ijm,abm,m -> iajb",M_ijm,M_abm,inv_eig,optimize='optimal'))
    W_ijba_correction = -2*(np.einsum("iam,jbm,m -> ijba",M_iam,M_jbm,inv_eig,optimize='optimal'))

    # Using W, find new X and Y
    gwe_diff = vir_gwe.reshape(-1,1) - core_gwe
    AW = -np.einsum("iabj->iajb",ovvo_anti,optimize='optimal') - W_iajb_correction
    AW += np.einsum("ai,ab,ij-> iajb", gwe_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')
    BW = -np.einsum("ijab->iajb",oovv_anti,optimize='optimal') - np.einsum("ijba->iajb",W_ijba_correction,optimize='optimal')

    eigW, XW, YW = bse.super_matrix_solver(AW,BW)
    print("GWE-BSE Complete")

    t_coeffic = YW@np.linalg.inv(XW)
    t2_reshaped = t2.reshape((n_occ*n_vir,n_occ*n_vir))

    np.savetxt("rpa_coeffic.csv",t_coeffic,delimiter=",")
    np.savetxt("bccd_coeffic.csv",t2_reshaped,delimiter=",")

    print("RPA eigenvals: ", np.sort(eig))
    print("GW-BSE eigenvals: ", np.sort(eigW))
    

