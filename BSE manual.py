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
_, eri_ao = spinor_one_and_two_e_int(mol)                                           # Find eri in spinor form
eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation

# Build anti-symmetrised integrals:
ijab = np.einsum("pi,qj,pqrs,ra,sb->ijab",core_spinorbs,core_spinorbs,eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal")
# <ij||ab> = <ij|ab> - <ij|ba>
ij_anti_ab = ijab - np.einsum("pi,qj,pqrs,rb,sa->ijab", core_spinorbs, core_spinorbs, eri_ao, vir_spinorbs, vir_spinorbs, optimize="optimal")

iabj =  np.einsum("pi,qa,pqrs,rb,sj->iabj",core_spinorbs,vir_spinorbs,eri_ao,vir_spinorbs,core_spinorbs,optimize="optimal")
# <ia||bj> = <ia|bj> - <ia|jb>
ia_anti_bj = iabj - np.einsum("pi,qa,pqrs,sj,rb->iabj", core_spinorbs, vir_spinorbs, eri_ao, core_spinorbs, vir_spinorbs, optimize="optimal")

iajb =  np.einsum("pi,qa,pqrs,rj,sb->iajb",core_spinorbs,vir_spinorbs,eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal")
# <ia||bj> = <ia|bj> - <ia|jb>
ia_anti_jb = iajb - np.einsum("pi,qa,pqrs,sb,rj->iajb", core_spinorbs, vir_spinorbs, eri_ao, vir_spinorbs, core_spinorbs,optimize="optimal")

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

# Remember to construct <ij||ab> and <ia||bj> (Given as exercise due to my laziness)

# Solve RPA equation to get W
# construct A and B and use supermatrix solver to get eigenvectors and values

A = np.einsum("iabj->iajb",ia_anti_bj)
B = np.einsum("ijab->iajb",ij_anti_ab)

for i in range(n_occ):
    for a in range(n_vir):
        A[i,a,i,a] += (vir_e[a]- core_e[i])

eig, X, Y = super_matrix_solver(A,B)

m_len = eig.shape[0]
X = X.reshape((n_occ,n_vir,m_len))
Y = Y.reshape((n_occ,n_vir,m_len))
print('RPA COMPLETE')

# Build W
W_iajb = ia_anti_jb
W_ijab = ij_anti_ab

# Build the required antisymmetrised orbitals
icjk =  np.einsum("pi,qc,pqrs,rj,sk->icjk",core_spinorbs,vir_spinorbs,eri_ao,core_spinorbs,core_spinorbs,optimize="optimal")
ic_anti_jk = icjk - np.einsum("pi,qc,pqrs,sk,rj->icjk", core_spinorbs,vir_spinorbs,eri_ao,core_spinorbs,core_spinorbs,optimize="optimal")

ikjc =  np.einsum("pi,qk,pqrs,rj,sc->ikjc",core_spinorbs,core_spinorbs,eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal")
ik_anti_jc = ikjc - np.einsum("pi,qk,pqrs,sc,rj->ikjc", core_spinorbs,core_spinorbs,eri_ao,vir_spinorbs,core_spinorbs,optimize="optimal")

acbk =  np.einsum("pa,qc,pqrs,rb,sk->acbk",vir_spinorbs,vir_spinorbs,eri_ao,vir_spinorbs,core_spinorbs,optimize="optimal")
ac_anti_bk = acbk - np.einsum("pa,qc,pqrs,sk,rb->acbk",vir_spinorbs,vir_spinorbs,eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal")

akbc =  np.einsum("pa,qk,pqrs,rb,sc->akbc",vir_spinorbs,core_spinorbs,eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal")
ak_anti_bc = akbc - np.einsum("pa,qk,pqrs,sc,rb->akbc",vir_spinorbs,core_spinorbs,eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal")

#Build the transfer/... coefficients

for i in range(n_occ):
    for a in range(n_vir):
        for j in range(n_occ):
            for b in range(n_vir):

                for m in range(m_len):
                    M_ijm = 0
                    M_abm = 0
                    M_aim = 0
                    M_jbm = 0
                    iajb_term = 0
                    ijab_term = 0
                    for k in range(n_occ):
                        for c in range(n_vir):
                            M_ijm = ic_anti_jk[i,c,j,k]*X[k,c,m] + ik_anti_jc[i,k,j,c]*Y[k,c,m]
                            M_abm = ac_anti_bk[a,c,b,k]*X[k,c,m] + ak_anti_bc[a,k,b,c]*Y[k,c,m]
                            M_aim = ia_anti_bj[i,c,a,k]*X[k,c,m] + ij_anti_ab[i,k,a,c]*Y[k,c,m]
                            M_jbm = ia_anti_bj[j,c,b,k]*X[k,c,m] + ij_anti_ab[j,k,b,c]*Y[k,c,m]

                            iajb_term += (M_ijm*M_abm) /eig[m]
                            ijab_term += (M_aim*M_jbm) /eig[m]

                    W_iajb[i,a,j,b] -= 2*iajb_term
                    W_ijab[i,j,a,b] -= 2*ijab_term

# Using W, find new X and Y
AW = np.einsum("iabj->iajb",iabj,optimize='optimal') - W_iajb
BW = np.einsum("ijab->iajb",ijab,optimize='optimal') - np.einsum("ijab->iajb",W_ijab,optimize='optimal')
# BW = np.zeros((n_occ,n_vir,n_occ,n_vir))

for i in range(n_occ):
    for a in range(n_vir):
        A[i,a,i,a] += (vir_gwe[a]- core_gwe[i])

eigW, XW, YW = super_matrix_solver(AW,BW)


t_coeffic = YW@np.linalg.inv(XW)
np.savetxt("coeffic.csv",t_coeffic,delimiter=",")

