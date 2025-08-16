import numpy as np
from pyscf import gto
from BSE_Helper import super_matrix_solver

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493


mol = gto.M(atom="Be 0.00 0.00 0.00",
            basis='aug-cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr")

# mol = gto.M(
#     atom = """N 0.12804615 0.00000000 0.00000000
#               H -0.59303935 0.88580079 -1.53425197
#               H -0.59303935 -1.77160157 0.00000000
#               H -0.59303935 0.88580079 1.53425197""",
#     basis = "aug-cc-pVTZ",  
#     spin = 0,
#     symmetry = False,
#     unit="Bohr",
# )
nocc = mol.nelectron // 2
myhf = mol.RHF.run()
mo = myhf.mo_coeff

# Core and Virtual Orbitals (spatial orbital basis)
o = mo[:, :nocc]
v = mo[:, nocc:]

nc = o.shape[1]
nv = v.shape[1]

# Constructing <ij|ab> and <ia|bj>
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
eri_ao = mol.intor('int2e')

# Orbital energies
core_e = myhf.mo_energy[:nocc]
vir_e = myhf.mo_energy[nocc:]
de = (vir_e.reshape(-1, 1) - core_e)

# Solve RPA equation
A_iajb = np.einsum("ai,ab,ij->iajb",de,np.identity(nv),np.identity(nc),optimize="optimal")
A_iajb += 2 * np.einsum("pi,qa,pqrs,rj,sb->iajb",o,v,eri_ao,o,v,optimize="optimal")
A_iajb += -np.einsum("pi,qj,pqrs,ra,sb->iajb",o,o,eri_ao,v,v,optimize="optimal")
B_iajb = 2 * np.einsum("pi,qa,pqrs,rb,sj->iajb",o,v,eri_ao,v,o,optimize="optimal")
B_iajb += -np.einsum("pi,qb,pqrs,ra,sj->iajb",o,v,eri_ao,v,o,optimize="optimal")

e_rpa, X_rpa, Y_rpa = super_matrix_solver(A_iajb, B_iajb)
print(np.sort(e_rpa)/eV_to_Hartree)