import numpy as np
from pyscf import gto, scf, cc

import src.BSE_Helper as helper
from src.cc_BSE_spatialorb import CC_BSE_spinfree
from src.cc_BSE_spinorb import CC_BSE_spin
from src.RPA_spatial import RPA_ORCA_all, RPA_spatial66
from src.RPA import RPA
from src.GW_BSE import GW_BSE

np.set_printoptions(precision=10, suppress=True, linewidth=100000)
eV2au = 0.0367493


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

print()
print(f'nocc:{n_occ_spatial}, nvir:{n_vir_spatial}')
print()

#spin-free
selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, se_occ_spa, se_vir_spa, hbse_eig, singE_spa, tripE_spa = \
 CC_BSE_spinfree(mol,mo,myhf,mycc,t2,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin-free basis COMPLETED.')
print()

#spin
hbse_sing, hbse_trip, selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, se_occ_spin, se_vir_spin, val, singE_spin, tripE_spin= \
  CC_BSE_spin(mol,mo,myhf,mycc,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin basis COMPLETED.')
print()

#RPA calculations
print('Standard RPA calculation.')
singE, tripE, rpa_eig, X_rpa, Y_rpa, A, B = RPA(mol,myhf,n_occ_spatial,n_vir_spatial)
print()
print(np.sort(singE)[:5]/eV2au)
print()
print('Standard RPA calculation COMPLETED.')
print()

# GW-BSE calculations from RPA
print("Starting GW-BSE calculation")
singEspa = GW_BSE(mol,myhf,X_rpa,Y_rpa,rpa_eig,n_occ_spatial)
print(np.sort(singEspa)/eV2au)
print('GW-BSE calculation COMPLETED.')
print()

print()
print('Starting Orca RPA calculation; SingEner:')
singEORCA, horca, A, B, _, _ ,t2_rpa = RPA_ORCA_all(mol,myhf,n_occ_spatial)
print(singEORCA[:5]/eV2au)
print()

print()
print('Starting RPA in Chris paper; SingEner, TripEner:')
singEPaper, tripEPaper, hrpa66 = RPA_spatial66(mol,myhf,n_occ_spatial,A,B,t2_rpa)
print(singEPaper[:5]/eV2au)
print(tripEPaper[:5]/eV2au)
print()


# with open("results.txt", "a", encoding="utf-8") as f:
#     f.write(f"{label}\n")
    
#     f.write('CC-BSE in spin-free basis; SingEner, TripEner:\n')
    
#     f.write(f"{singE_spa[:5]/eV2au}\n")
#     f.write(f"{tripE_spa[:5]/eV2au}\n")
#     f.write("\n")

#     f.write('Starting Orca RPA calculation; SingEner:\n')
#     singEORCA, horca, A, B, _, _, t2_rpa = RPA_ORCA_all(mol,myhf,n_occ_spatial)
#     f.write(f"{singEORCA[:5]/eV2au}\n")
#     f.write("\n")

#     f.write('Starting RPA in Chris paper; SingEner, TripEner:\n')
#     singEPaper, tripEPaper, hrpa66 = RPA_spatial66(mol,myhf,n_occ_spatial,A,B,t2_rpa)
#     f.write(f"{singEPaper[:5]/eV2au}\n")
#     f.write(f"{tripEPaper[:5]/eV2au}\n")

#     diff = RPAselfConsisCheck(horca, hrpa66)
#     f.write(f"RPA Self-Consistency Check: {diff}")
#     f.write("\n\n")










































