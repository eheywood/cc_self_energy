import numpy as np
from pyscf import gto
import BSE_Helper as helper
from scipy.linalg import block_diag

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def get_RPA_amps(vir_e, core_e, ovvo_anti, oovv_anti, n_occ,n_vir) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # construct A and B and use supermatrix solver to get eigenvectors and values
    e_diff = vir_e.reshape(-1,1) - core_e
    A = -np.einsum("iabj->iajb", ovvo_anti,optimize='optimal') 
    A += np.einsum("ai,ab,ij-> iajb", e_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')

    B = np.einsum("ijab->iajb", oovv_anti,optimize='optimal')

    #print(A.shape)
    #print(B.shape)

    eig, X, Y = helper.super_matrix_solver(A,B)
    #print('RPA COMPLETE')
    return eig, X, Y, A, B

def build_RPA_hamiltonian(vir_e, core_e, ovvo_anti, oovv_anti,n_occ,n_vir,t2) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    
    H_rpa = np.zeros((n_occ,n_vir,n_occ,n_vir))
    e_diff = vir_e.reshape(-1,1) - core_e

    term_1 = np.einsum("ai,ab,ij->iajb", e_diff,np.identity(n_vir),np.identity(n_occ),optimize='optimal')
    term_2 = -np.einsum("iabj->iajb", ovvo_anti,optimize='optimal')

    term_3 = np.einsum("ikbc,jkca->ibja",oovv_anti,t2, optimize='optimal')

    H_rpa = term_1 + term_2 + term_3

    return H_rpa, term_1, term_2, term_3

def RPA(mol, myhf, n_occ_spatial, n_vir_spatial) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Run RPA on a molecule to get the eigenvalues and eigenvectors of the RPA equation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule to run RPA on.
    myhf : pyscf.scf.hf.RHF
        The Hartree-Fock object for the molecule.
    n_occ_spatial : int
        Number of occupied spatial orbitals.
    n_vir_spatial : int
        Number of virtual spatial orbitals.
    eV2au : float, optional
        Conversion factor from eV to Hartree, by default 0.0367493

    Returns
    -------
    tuple[np.ndarray,np.ndarray,np.ndarray]
        singlet excitations,triplet excitations
    """
    
    mo = myhf.mo_coeff
    core_spinorbs, vir_spinorbs = helper.get_spinorbs(mo, n_occ_spatial)

    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    # Orbital energies
    core_e = np.array(list(myhf.mo_energy[:n_occ_spatial]) + list(myhf.mo_energy[:n_occ_spatial]))
    vir_e = np.array(list(myhf.mo_energy[n_occ_spatial:]) + list(myhf.mo_energy[n_occ_spatial:]))

    # Constructing <ij|ab> and <ia|bj>
    _, eri_ao = helper.spinor_one_and_two_e_int(mol)                   # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # Build the required anti-symmetrised orbitals
    oovv_anti,_,_,ovvo_anti,_ = helper.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # Solve RPA equations to get amplitudes
    rpa_eig, X_rpa, Y_rpa, A, B = get_RPA_amps(vir_e, core_e, ovvo_anti, oovv_anti, n_occ,n_vir)

    nA = int(np.sqrt(A.size))
    nB = int(np.sqrt(B.size))
    A = A.reshape((nA, nA))
    B = B.reshape((nB, nB))

    t = Y_rpa@np.linalg.inv(X_rpa)
    t_reshaped = t.reshape((n_occ,n_vir,n_occ,n_vir))
    t_reshaped = np.einsum("iajb->ijba",t_reshaped,optimize='optimal')

    H_rpa,term1,term2,term3 = build_RPA_hamiltonian(vir_e,core_e,ovvo_anti,oovv_anti,n_occ,n_vir,t_reshaped)

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    hrpa_sing = helper.sing_excitation(H_rpa, n_occ_spatial, n_vir_spatial)
    hrpa_trip = helper.trip_excitation(H_rpa, n_occ_spatial, n_vir_spatial)

    singE, _ = np.linalg.eig(hrpa_sing)
    tripE, _ = np.linalg.eig(hrpa_trip)

    #assert np.allclose(np.imag(singE), np.zeros(singE.shape), rtol=0, atol=1e-8)    # Real eigenvalues
    singE = np.real(singE)
    #assert np.allclose(np.imag(tripE), np.zeros(tripE.shape), rtol=0, atol=1e-8)    # Real eigenvalues
    tripE = np.real(singE)
    
    H_rpa = H_rpa.reshape((n_occ*n_vir, n_occ*n_vir))
    H_rpa_mat = A + B@t

    ## DEBUGGING SELF CONSISTENCY CHECKS ##
    print("A check")
    print(np.average(np.absolute((np.sort(A.reshape(n_occ,n_vir,n_occ,n_vir))-np.sort(term1+term2)))))

    print("B check")
    print(np.average(np.absolute((np.sort((B@t).reshape(n_occ,n_vir,n_occ,n_vir))-np.sort(term3)))))

    # print(B.shape)
    # print(t.shape)
    # print((B@t).shape)

    diff = np.average(np.absolute(np.sort((H_rpa_mat).reshape(n_occ*n_vir,n_occ*n_vir))-np.sort(H_rpa)))
    print(f'RPA Self-consistency check: {diff}')

    return singE, tripE, rpa_eig, X_rpa, Y_rpa


