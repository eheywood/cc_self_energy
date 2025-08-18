import pytest
from pyscf import gto,cc
import numpy as np

from src.cc_BSE_spatialorb import CC_BSE_spinfree
from src.cc_BSE_spinorb import CC_BSE_spin

hartree_ev = 27.2114

@pytest.fixture(scope="class", params=[
    ("H 0.00 0.00 0.00; H 0.00 0.00 2.00", 'aug-cc-pVTZ', "H2")
])
def setup_helper(request):
    geom, basis, label = request.param
    class Data:
        pass
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
    return data

@pytest.mark.usefixtures("setup_helper")
class Test_HBSE_consistency:

    def test_HBSE_spin_spat_consistency(self, setup_helper):
        data = setup_helper
        hartree_ev = 27.2114

        #spin-free
        hbse_0,selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, se_occ_spa, se_vir_spa, hbse_v_spa, singEspa, tripEspa = \
        CC_BSE_spinfree(data.mol,
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

        #spin
        selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, se_occ_spin, se_vir_spin, hbse_v_spin, singE_spin, tripE_spin = \
        CC_BSE_spin(data.mol,
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

        print(np.sort(singEspa)[:5])
        print(np.sort(singE_spin)[:5])

        assert singE_spin[3] == singEspa[4]
        assert np.sort(singEspa).all() == np.sort(singE_spin).all(), "Singlet excitation energies do not match between spin-free and spin basis."
        assert np.sort(tripEspa).all() == np.sort(tripE_spin).all(), "Triplet excitation energies do not match between spin-free and spin basis."


