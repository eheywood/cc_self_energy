import pytest
from pyscf import gto,cc
import numpy as np
from scipy.linalg import schur

from src.cc_RPA import RPA
from src.spatial_RPA import RPA_spatial
from src.GW_BSE import GW_BSE
from tests.test_helper import count_matches, write_to_file,molecules

hartree_ev = 27.2114

class Data:
    label:str
    mol:gto.Mole
    myhf = None
    mycc = None
    mo = None
    t2 = None
    n_occ_spatial:int
    n_vir_spatial:int
    n_occ_spin:int
    n_vir_spin:int   
    pass

class RPA_Results:
    singE: np.ndarray
    tripE: np.ndarray
    eig: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    B: np.ndarray
    pass

class GW_BSE_Results:
    pass

@pytest.fixture(scope="class", params=molecules)
def setup_helper(request):
    """Fixture to set up the molecule and perform CC-BSE calculations."""

    # Begin calculation and unpack parameters
    geom, basis, label = request.param
    data = Data()
    data.label = label
    data.mol = gto.M(atom=geom,
                basis=basis,
                spin=0,
                symmetry=False,
                unit="Bohr")
    data.myhf = data.mol.HF.run() 
    data.mo = data.myhf.mo_coeff

    n_mo = data.myhf.mo_coeff.shape[1]
    data.n_occ_spatial = data.mol.nelec[0]  # Number of occupied orbitals (alpha electrons in RHF)
    data.n_vir_spatial = n_mo - data.n_occ_spatial

    # RPA CALCULATION SPIN ORBITAL
    print('Standard RPA calculation.')
    singEspa, tripEspa, rpa_eig, rpa_X, rpa_Y, A, B = RPA(data.mol,data.myhf,data.n_occ_spatial,data.n_vir_spatial)
    print()
    print('Standard RPA calculation COMPLETED.')
    
    RPA_results = RPA_Results()
    RPA_results.singE = singEspa
    RPA_results.tripE = tripEspa
    RPA_results.eig = rpa_eig
    RPA_results.X = rpa_X
    RPA_results.Y = rpa_Y
    RPA_results.A = A
    RPA_results.B = B

    # GW_BSE CALCULATION
    gwbse_sing, _ , _ = GW_BSE(data.mol,data.myhf,RPA_results.X,RPA_results.Y,RPA_results.eig,data.n_occ_spatial)

    # SPATIAL RPA CALCULATION
    rpa_spatial_singlets = RPA_spatial(data.mol,data.myhf,data.n_occ_spatial)
  
    # prep results file
    title = f"RPA Self-Consistency and GW-BSE Results for {data.label} (eV)"
    write_to_file(data.label, title,heading=True)

    yield data, RPA_results

    # Cleanup after tests

    # write first 5 excitation energies to file
    write_to_file(data.label, f"\nSpin-Orbital RPA Excitation Predictions\n")

    singlet_str_spin = "    Singlet excitation energies: " + str(np.sort(RPA_results.singE)[:5]*hartree_ev)
    triplet_str_spin = "    Triplet excitation energies: " + str(np.sort(RPA_results.tripE)[:5]*hartree_ev)
    write_to_file(data.label, singlet_str_spin)
    write_to_file(data.label, triplet_str_spin)

    write_to_file(data.label, f"\nSpin-Free RPA Excitation Predictions \n")

    singlet_str_spa = "    Singlet excitation energies: " + str(np.sort(rpa_spatial_singlets)[:5]*hartree_ev)
    write_to_file(data.label, singlet_str_spa)

    write_to_file(data.label, f"\nGW-BSE Excitation Predictions \n")

    gwbse_str = "    GW-BSE Excitation energies: " + str(np.sort(gwbse_sing)[:5]*hartree_ev)
    write_to_file(data.label, gwbse_str)

    write_to_file(data.label, "")

@pytest.mark.usefixtures("setup_helper")
class Test_RPA_n_GW_BSE:
    
    def test_spin_RPA_ham(self, setup_helper):
        data, RPA_spin_results = setup_helper

        t = RPA_spin_results.Y@RPA_spin_results.X.T

        H_rpa_mat = RPA_spin_results.A + RPA_spin_results.B@t

        t,z = schur(H_rpa_mat)

        if np.triu(t).all() == t.all():
            e = np.diag(t)
        else:
            quit("Schur decomposition failed, matrix is not upper triangular")
        v = z
        

        numer, denom, label_str = count_matches(RPA_spin_results.eig,  e, "RPA Eigenvalues")
        write_to_file(data.label, label_str)

        assert numer == denom, f"RPA Hamiltonian does not match eigenvalues from RPA Matrix calculation. {numer}/{denom} matched."
        
    
        



