import numpy as np
from pyscf import gto
from scipy.linalg import block_diag
from BSE_Helper import spinor_one_and_two_e_int, super_matrix_solver

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
            basis='cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr")
myhf = mol.RHF.run() 
mo = myhf.mo_coeff
print(mo.shape)

# Core and Virtual Orbitals (spatial orbital basis)
core_spatialorbs = mo[:, 0].reshape(-1,1)
vir_spatialorbs = mo[:, 1:]

# Convert core and virtual orbitals into spin-orbital form. The first half of the columns will be alpha spin orbs, the
# next half of the columns will be beta spin orbs
core_spinorbs = block_diag(core_spatialorbs, core_spatialorbs)
vir_spinorbs = block_diag(vir_spatialorbs, vir_spatialorbs)

n_occ = core_spinorbs.shape[1]
n_vir = vir_spinorbs.shape[1]
print(f"n_occ = {n_occ}, n_vir = {n_vir}")

# Constructing <ij|ab> and <ia|bj>
# https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
_, eri_ao = spinor_one_and_two_e_int(mol)                       # Find eri in spinor form
eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

# <ij|ab>
ijab = np.einsum("pi,qj,pqrs,ra,sb->ijab",core_spinorbs,core_spinorbs,eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal")
#<ia|bj>
iabj =  np.einsum("pi,qa,pqrs,rb,sj->iabj",core_spinorbs,vir_spinorbs,eri_ao,vir_spinorbs,core_spinorbs,optimize="optimal")
#<ia|bj>
iajb =  np.einsum("pi,qa,pqrs,rj,sb->iajb",core_spinorbs,vir_spinorbs,eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal")

# Build the required anti-symmetrised orbitals
oovv_anti = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spinorbs, core_spinorbs, anti_eri_ao, vir_spinorbs, vir_spinorbs, optimize="optimal")
ooov_anti =  np.einsum("pi,qk,pqrs,rj,sc->ikjc",core_spinorbs,core_spinorbs,anti_eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal") 
vovv_anti =  np.einsum("pa,qk,pqrs,rb,sc->akbc",vir_spinorbs,core_spinorbs,anti_eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal") 
ovvo_anti = np.einsum("pi,qa,pqrs,rb,sj->iabj", core_spinorbs, vir_spinorbs, anti_eri_ao, vir_spinorbs, core_spinorbs,optimize="optimal") 
ovov_anti = np.einsum("pi,qa,pqrs,rj,sb->iajb", core_spinorbs, vir_spinorbs, anti_eri_ao, core_spinorbs, vir_spinorbs, optimize="optimal")

# Orbital energies
core_e = np.array(list(myhf.mo_energy[:1]) + list(myhf.mo_energy[:1]))
vir_e = np.array(list(myhf.mo_energy[1:]) + list(myhf.mo_energy[1:]))
# Self energies (in eV)
core_gwe_ev = [-15.12445, -15.12445]
vir_gwe_ev = [3.5165,9.17075,14.79584,18.41143,18.41143,26.94759,28.61681,28.61681,39.95175,69.342,72.37157,86.87528,
         86.87528,87.97988,87.97988,92.39739,102.77003,102.77003,106.97273,106.97273,109.12804,109.12804,110.80976,
         129.84903,129.84903,132.26775,166.48888, 3.5165,9.17075,14.79584,18.41143,18.41143,26.94759,28.61681,28.61681,39.95175,69.342,72.37157,86.87528,
         86.87528,87.97988,87.97988,92.39739,102.77003,102.77003,106.97273,106.97273,109.12804,109.12804,110.80976,
         129.84903,129.84903,132.26775,166.48888]
# Self energies (in atomic units)
core_gwe = eV_to_Hartree * np.array(core_gwe_ev)
vir_gwe = eV_to_Hartree * np.array(vir_gwe_ev)


# Solve RPA equation to get W
# construct A and B and use supermatrix solver to get eigenvectors and values
e_diff = vir_e.reshape(-1,1) - core_e
A = -np.einsum("iabj->iajb", ovvo_anti) 
A += np.einsum("ai,ab,ij-> iajb", e_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')
B = -np.einsum("ijab->iajb", oovv_anti)


eig, X, Y = super_matrix_solver(A,B)

m_len = eig.shape[0]
X = X.reshape((n_occ,n_vir,m_len))
Y = Y.reshape((n_occ,n_vir,m_len))
print('RPA COMPLETE')

# Build W
#Build the transfer/... coefficients
M_ijm = np.einsum("ikjc,kcm->ijm",ooov_anti,X+Y,optimize='optimal')
# print("M_ijm: ", max(np.abs(X.reshape(-1))))
M_abm = np.einsum("akbc,kcm->abm",vovv_anti,X+Y,optimize='optimal') 
# print("M_abm: ", max(np.abs(Y.reshape(-1))))
M_iam = np.einsum("ikac,kcm->iam",oovv_anti,X+Y,optimize='optimal') 
M_jbm = M_iam

inv_eig = 1/eig
W_iajb_correction = -2*(np.einsum("ijm,abm,m -> iajb",M_ijm,M_abm,inv_eig,optimize='optimal'))
W_ijba_correction = -2*(np.einsum("iam,jbm,m -> ijba",M_iam,M_jbm,inv_eig,optimize='optimal'))
print(inv_eig)
# Using W, find new X and Y

gwe_diff = vir_gwe.reshape(-1,1) - core_gwe
AW = -np.einsum("iabj->iajb",ovvo_anti,optimize='optimal') - W_iajb_correction
AW += np.einsum("ai,ab,ij-> iajb", gwe_diff, np.identity(n_vir),np.identity(n_occ),optimize='optimal')
BW = -np.einsum("ijab->iajb",oovv_anti,optimize='optimal') - np.einsum("ijba->iajb",W_ijba_correction,optimize='optimal')

print("W_iajb_corr: ", - W_iajb_correction[0,0,0,0])

# print("AW, BW: ",AW[0,0,0,0], BW[0,0,0,0])
# print("Wiajb corr, Wijab corr: ", W_iajb_correction[0,0,0,0], W_ijba_correction[0,0,0,0])

eigW, XW, YW = super_matrix_solver(AW,BW)
print("GWE-BSE Complete")

t_coeffic = YW@np.linalg.inv(XW)
print("Coefficients shape: ", t_coeffic.shape)

np.savetxt("coeffic.csv",t_coeffic,delimiter=",")
print(max(np.abs(t_coeffic.reshape(-1))))

