import numpy as np
from pyscf import gto, scf, cc
import BSE_Helper as helper
from cc_BSE_spatialorb import CC_BSE_spinfree
from cc_BSE_spinorb import CC_BSE_spin
from cc_RPA import RPA
from spatial_RPA import RPA_spatial
from GW_BSE import GW_BSE

np.set_printoptions(precision=10, suppress=True, linewidth=100000)
eV2au = 0.0367493


##Define Molecule to calculate amplitudes and mo for
# label = 'H2'
# mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
#           basis='cc-pVDZ',
#           spin=0,
#           symmetry=False,
#           unit="Bohr")

#label = 'CH3CHO'
#mol = gto.M(
#    atom = """C -0.00234503 0.00000000 0.87125063
#            C -1.75847785 0.00000000 -1.34973671
#            O  2.27947397 0.00000000 0.71968028
#            H -0.92904537 0.00000000 2.73929404
#            H -2.97955463 1.66046488 -1.25209463
#            H -2.97955463 -1.66046488 -1.25209463
#            H -0.70043433 0.00000000 -3.11066412""",
#    basis = "aug-cc-pVTZ",  
#    spin = 0,
#    symmetry = False,
#    unit = "Bohr")

#label = 'NH3'
#mol = gto.M(
#atom = """N 0.12804615 0.00000000 0.00000000
#          H -0.59303935 0.88580079 -1.53425197
#          H -0.59303935 -1.77160157 0.00000000
#          H -0.59303935 0.88580079 1.53425197""",
#basis = "aug-cc-pVTZ",  
#spin = 0,
#symmetry = False,
#unit="Bohr")

label = 'Be'
mol = gto.M(atom="Be 0.00000000 0.00000000 0.00000000",
            basis='aug-cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr")

print()
print(label)
print()

myhf = mol.HF.run() 
mycc = cc.BCCD(myhf,max_cycle = 200, conv_tol_normu=1e-8).run()
mo = mycc.mo_coeff
t2 = mycc.t2

n_occ_spatial = int(t2.shape[0])
n_vir_spatial = int(t2.shape[2])
n_occ_spin = int(t2.shape[0]*2)
n_vir_spin = int(t2.shape[2]*2)
#print('bcc mo coeff')
#print(mo[:, 0])

print()
print(f'nocc:{n_occ_spatial}, nvir:{n_vir_spatial}')
print()

#spin-free
term1_spa, term2_spa, hbse_0,selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, se_occ_spa, se_vir_spa, hbse_v_spa, singEspa, tripEspa = \
 CC_BSE_spinfree(mol,mo,myhf,mycc,t2,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin-free basis COMPLETED.')
print()

#spin
term1_spin, term2_spin,selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, se_occ_spin, se_vir_spin, hbse_v_spin, singEspin, tripEspin = \
  CC_BSE_spin(mol,mo,myhf,mycc,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin basis COMPLETED.')
print()

##debugging############################################################
## Correction term check - printing the matrix for Sigma
# print()
# print('CORRECTION MATRIX CHECK')
# print(f'selfener_occ_spa:{selfener_occ_spa}')
# print(f'selfener_vir_spa:{selfener_vir_spa}')
# print(f'selfener_occ_spin:{selfener_occ_spin}')
# print(f'selfener_vir_spin:{selfener_vir_spin}')
# print()

## Fock matrix check - printing the fock matrix
# print()
# print('FOCK MATRIX CHECK')
# print(f'fock_occ_spa:{np.diag(fock_occ_spa)[:10]/eV2au}')
# print(f'fock_vir_spa:{np.diag(fock_vir_spa)[:10]/eV2au}')
# print(f'fock_occ_spin:{fock_occ_spin}')
# print(f'fock_vir_spin:{fock_vir_spin}')
# print()

## Self energy matrix check - printing the self energy matrix
# print()
# print('SELF ENERGY MATRIX CHECK')
# print(f'se_occ_spa:{se_occ_spa}')
# print(f'se_vir_spa:{se_vir_spa}')
# print(f'se_occ_spin:{se_occ_spin}')
# print(f'se_vir_spin:{se_vir_spin}')
# print()

# Comparing the matrix
print()
helper.count_matches(selfener_occ_spa,  selfener_occ_spin, "Corr. occ")
helper.count_matches(selfener_vir_spa,  selfener_vir_spin, "Corr. vir")
helper.count_matches(fock_occ_spa,  fock_occ_spin, "Fock  occ")
helper.count_matches(fock_vir_spa,  fock_vir_spin, "Fock  vir")
helper.count_matches(se_occ_spa,  se_occ_spin, "SelfE occ")
helper.count_matches(se_vir_spa,  se_vir_spin, "SelfE vir")
print()
##########################################################################


np.savetxt("results_hbse_spa.txt", np.sort(term1_spa))
np.savetxt("results_hbse_spin.txt", np.sort(term1_spin))



helper.count_matches(term1_spa, term1_spin, "<ia||bj>")

#helper.count_matches(singEspa, singEspin, "<ia||bj>")
#helper.count_matches(tripEspa, tripEspin, "<ia||bj>")
helper.count_matches(term2_spa, term2_spin, "<ik||bc>t")
helper.count_matches(hbse_v_spa, hbse_v_spin, "hbse")

# chunk = 1000
# for start in range(0, n, chunk):
#     end = min(start + chunk, n)
#     print(f"chunk {start}:{end}")
#     helper.count_matches(hbse_v_spa[start:end], hbse_v_spin, "hbse")


# #RPA calculations
# print('Standard RPA calculation.')
# singEspa, tripEspa, rpa_eig, rpa_X, rpa_Y = RPA(mol,myhf,n_occ_spatial,n_vir_spatial)
# print()
# print(np.sort(singEspa)[:5]/eV2au)
# print()
# print('Standard RPA calculation COMPLETED.')
# print()

# GW-BSE calculations from RPA
#print("Starting GW-BSE calculation")
#singEspa = GW_BSE(mol,myhf,rpa_X,rpa_Y,rpa_eig,n_occ_spatial)
#print(np.sort(singEspa)/eV2au)
#print('GW-BSE calculation COMPLETED.')
# print()

# print()
# print('Starting Orca RPA calculation.')
# print()
# singEspa = RPA_spatial(mol,myhf,n_occ_spatial)
# print(np.sort(singEspa)[:5]/eV2au)
# print()
# print('Orca RPA calculation COMPLETED')
# print()





































