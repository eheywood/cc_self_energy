import numpy as np
from pyscf import gto
from BSE_Helper import super_matrix_solver

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def RPA_spatial(mol,myhf,n_occ) -> np.ndarray:
  
  mo = myhf.mo_coeff

  print(n_occ)
  
  # Core and Virtual Orbitals (spatial orbital basis)
  o = mo[:, :n_occ]
  v = mo[:, n_occ:]

  nc = o.shape[1]
  nv = v.shape[1]

  # Constructing <ij|ab> and <ia|bj>
  # hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
  eri_ao = mol.intor('int2e')

  # Orbital energies
  core_e = myhf.mo_energy[:n_occ]
  vir_e = myhf.mo_energy[n_occ:]
  de = (vir_e.reshape(-1, 1) - core_e)

  # Solve RPA equation (ORCA Style)
  A_iajb = np.einsum("ai,ab,ij->iajb",de,np.identity(nv),np.identity(nc),optimize="optimal")
  A_iajb += 2 * np.einsum("pi,qa,pqrs,rj,sb->iajb",o,v,eri_ao,o,v,optimize="optimal")
  A_iajb += -np.einsum("pi,qj,pqrs,ra,sb->iajb",o,o,eri_ao,v,v,optimize="optimal")
  B_iajb = 2 * np.einsum("pi,qa,pqrs,rb,sj->iajb",o,v,eri_ao,v,o,optimize="optimal")
  B_iajb += -np.einsum("pi,qb,pqrs,ra,sj->iajb",o,v,eri_ao,v,o,optimize="optimal")

  e_rpa, _, _ = super_matrix_solver(A_iajb, B_iajb)
  print(np.sort(e_rpa)/eV_to_Hartree)

  return e_rpa