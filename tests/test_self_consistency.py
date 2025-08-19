import pytest
from pyscf import gto,cc
import numpy as np

from src.cc_BSE_spatialorb import CC_BSE_spinfree
from src.cc_BSE_spinorb import CC_BSE_spin
from tests.test_helper import count_matches

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

class BSE_results:
    hbse_0: np.ndarray
    selfener_occ: np.ndarray
    selfener_vir: np.ndarray
    fock_occ: np.ndarray
    fock_vir: np.ndarray
    se_occ: np.ndarray
    se_vir: np.ndarray
    hbse_v: np.ndarray
    singE: np.ndarray
    tripE: np.ndarray
    pass

@pytest.fixture(scope="class", params=[
    ("H 0.00 0.00 0.00; H 0.00 0.00 2.00", 'aug-cc-pVTZ', "H2"),
    ("Be 0.00000000 0.00000000 0.00000000", 'aug-cc-pVTZ', "Be"),
])
def setup_helper(request):
    geom, basis, label = request.param
    data = Data()
    data.label = label
    data.mol = gto.M(atom=geom,
                basis=basis,
                spin=0,
                symmetry=False,
                unit="Bohr")
    data.myhf = data.mol.HF.run() 
    data.mycc = cc.BCCD(data.myhf,max_cycle = 200, conv_tol_normu=1e-8).run()
    data.mo = data.mycc.mo_coeff
    data.t2 = data.mycc.t2
    data.n_occ_spatial = int(data.t2.shape[0])
    data.n_vir_spatial = int(data.t2.shape[2])
    data.n_occ_spin = int(data.t2.shape[0]*2)
    data.n_vir_spin = int(data.t2.shape[2]*2)

    #spin-free
    hbse_0_spin,selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, se_occ_spa, se_vir_spa, hbse_v_spa, singEspa, tripEspa = CC_BSE_spinfree(data.mol,
                                        data.mo,
                                        data.myhf,
                                        data.mycc,
                                        data.t2,
                                        data.label,
                                        1/hartree_ev,
                                        data.n_occ_spatial,
                                        data.n_vir_spatial,
                                        data.n_occ_spin,
                                        data.n_vir_spin)
    print()
    print('CC-BSE in spin-free basis COMPLETED.')
    print()

    spat_results = BSE_results()
    spat_results.hbse_0 = hbse_0_spin
    spat_results.selfener_occ = selfener_occ_spa
    spat_results.selfener_vir = selfener_vir_spa
    spat_results.fock_occ = fock_occ_spa
    spat_results.fock_vir = fock_vir_spa
    spat_results.se_occ = se_occ_spa
    spat_results.se_vir = se_vir_spa
    spat_results.hbse_v = hbse_v_spa
    spat_results.singE = singEspa
    spat_results.tripE = tripEspa

    #spin
    selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, se_occ_spin, se_vir_spin, hbse_v_spin, singE_spin, tripE_spin = CC_BSE_spin(data.mol,
                                        data.mo,
                                        data.myhf,
                                        data.mycc,
                                        data.t2,
                                        data.label,
                                        1/hartree_ev,
                                        data.n_occ_spatial,
                                        data.n_vir_spatial,
                                        data.n_occ_spin,
                                        data.n_vir_spin)
    print()
    print('CC-BSE in spin basis COMPLETED.')
    print()

    spin_results = BSE_results()
    spin_results.selfener_occ = selfener_occ_spin
    spin_results.selfener_vir = selfener_vir_spin
    spin_results.fock_occ = fock_occ_spin
    spin_results.fock_vir = fock_vir_spin
    spin_results.se_occ = se_occ_spin
    spin_results.se_vir = se_vir_spin
    spin_results.hbse_v = hbse_v_spin
    spin_results.singE = singE_spin
    spin_results.tripE = tripE_spin 

    return data, spat_results, spin_results

@pytest.mark.usefixtures("setup_helper")
class Test_HBSE_consistency:

    def test_HBSE_spin_spat_consistency(self, setup_helper):
        _, spat_results, spin_results = setup_helper

        # Count matches for Hamiltonian
        singlet_numer, singlet_denom = count_matches(spat_results.singE,  spin_results.singE, "Singlet excitation energies")
        triplet_numer, triplet_denom = count_matches(spat_results.tripE,  spin_results.tripE, "Triplet excitation energies")

        assert singlet_numer == singlet_denom, f"Singlet excitation energies do not match between spin-free and spin basis. {singlet_numer}/{singlet_denom} matched."
        assert triplet_numer == triplet_denom, f"Triplet excitation energies do not match between spin-free and spin basis. {triplet_numer}/{triplet_denom} matched."
    
    def test_self_energy_consistency(self, setup_helper):
        _, spat_results, spin_results = setup_helper

        # Count matches for Hamiltonian
        occ_numer, occ_denom = count_matches(spat_results.selfener_occ,  spin_results.selfener_occ, "Occupied self-energy")
        vir_numer, vir_denom = count_matches(spat_results.selfener_vir,  spin_results.selfener_vir, "Virtual self-energy")

        assert occ_numer == occ_denom, f"Occupied self-energy does not match between spin-free and spin basis. {occ_numer}/{occ_denom} matched."
        assert vir_numer == vir_denom, f"Virtual self-energy does not match between spin-free and spin basis. {vir_numer}/{vir_denom} matched."


