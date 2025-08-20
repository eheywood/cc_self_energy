import numpy as np
from pyscf import gto, scf, cc

import src.BSE_Helper as helper
from src.cc_BSE_spatialorb import CC_BSE_spinfree
from src.cc_BSE_spinorb import CC_BSE_spin
from src.cc_RPA_all import RPA_ORCA_all, RPA_spatial66, RPAselfConsisCheck
from src.GW_BSE import GW_BSE

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

# label = 'H2O'
# mol = gto.M(atom="O 0.00000000 0.00000000 -0.13209669; H 0.00000000 1.43152878 0.97970006; H 0.00000000 -1.43152878 0.97970006",
#             basis='cc-pVTZ',
#             spin=0,
#             symmetry=False,
#             unit="Bohr")

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
term1_diag_spa, term2_diag_spa, selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, se_occ_spa, se_vir_spa, hbse_eig, singE_spa, tripE_spa = \
 CC_BSE_spinfree(mol,mo,myhf,mycc,t2,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin-free basis COMPLETED.')
print()

#spin
hbse_sing, hbse_trip, term1_diag, term2_diag, selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, se_occ_spin, se_vir_spin, val, singE_spin, tripE_spin= \
  CC_BSE_spin(mol,mo,myhf,mycc,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin basis COMPLETED.')
print()

#debugging
# print(f'selfener_occ_spa:{selfener_occ_spa}')
# print(f'selfener_vir_spa:{selfener_vir_spa}')
# print(f'selfener_occ_spin:{selfener_occ_spin}')
# print(f'selfener_vir_spin:{selfener_vir_spin}')
# print()

# # Fock matrix check - printing the fock matrix
# print()
# print('FOCK MATRIX CHECK')
# print(f'fock_occ_spa:{np.diag(fock_occ_spa)[:10]/eV2au}')
# print(f'fock_vir_spa:{np.diag(fock_vir_spa)[:10]/eV2au}')

#print(f'fock_occ_spin:{fock_occ_spin}')
#print(f'fock_vir_spin:{fock_vir_spin}')

#print(se_occ_spa)
#print(se_vir_spa)
#print(se_occ_spin)
#print(se_vir_spin)

helper.count_matches(se_occ_spa,  se_occ_spin, "occ self-energy+fockener")
helper.count_matches(se_vir_spa,  se_vir_spin, "vir self-energy+fockener")
print()
# helper.count_matches(hbse_v_spa, hbse_v_spin, "hbse")


# np.savetxt("results_hbse[0]_spatial.txt", hbse_0.reshape(-1))


# #RPA calculations
# print('Standard RPA calculation.')
# singEspa, tripEspa, rpa_eig, rpa_X, rpa_Y = RPA(mol,myhf,n_occ_spatial,n_vir_spatial)
# print()
# print(np.sort(singEspa)[:5]/eV2au)
# print()
# print('Standard RPA calculation COMPLETED.')
# print()

#GW-BSE calculations from RPA
# print("Starting GW-BSE calculation")
# singEspa = GW_BSE(mol,myhf,rpa_X,rpa_Y,rpa_eig,n_occ_spatial)
# print(np.sort(singEspa)/eV2au)
# print('GW-BSE calculation COMPLETED.')
# print()

print()
print('Starting Orca RPA calculation; SingEner:')
singEORCA, horca, A, B, t2_rpa = RPA_ORCA_all(mol,myhf,n_occ_spatial)
print(singEORCA[:5]/eV2au)
print()

print()
print('Starting RPA in Chris paper; SingEner, TripEner:')
singEPaper, tripEPaper, hrpa66 = RPA_spatial66(mol,myhf,n_occ_spatial,A,B,t2_rpa)
print(singEPaper[:5]/eV2au)
print(tripEPaper[:5]/eV2au)
print()


print()
diff = RPAselfConsisCheck(horca, hrpa66)
print(f'RPA Self Consistency Check:{diff}')


with open("results.txt", "a", encoding="utf-8") as f:
    f.write(f"{label}\n")
    
    f.write('CC-BSE in spin-free basis; SingEner, TripEner:\n')
    
    f.write(f"{singE_spa[:5]/eV2au}\n")
    f.write(f"{tripE_spa[:5]/eV2au}\n")
    f.write("\n")

    f.write('Starting Orca RPA calculation; SingEner:\n')
    singEORCA, horca, A, B, t2_rpa = RPA_ORCA_all(mol,myhf,n_occ_spatial)
    f.write(f"{singEORCA[:5]/eV2au}\n")
    f.write("\n")

    f.write('Starting RPA in Chris paper; SingEner, TripEner:\n')
    singEPaper, tripEPaper, hrpa66 = RPA_spatial66(mol,myhf,n_occ_spatial,A,B,t2_rpa)
    f.write(f"{singEPaper[:5]/eV2au}\n")
    f.write(f"{tripEPaper[:5]/eV2au}\n")

    diff = RPAselfConsisCheck(horca, hrpa66)
    f.write(f"RPA Self-Consistency Check: {diff}")
    f.write("\n\n")










































