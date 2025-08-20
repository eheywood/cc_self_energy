import numpy as np
from pyscf import gto
import BSE_Helper as helper

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493

def RPA_ORCA(mol,myhf,n_occ) -> np.ndarray:
  
  mo = myhf.mo_coeff

  #print(n_occ)
  
  # Core and Virtual Orbitals (spatial orbital basis)
  o = mo[:, :n_occ]
  v = mo[:, n_occ:]

  nc = o.shape[1]
  nv = v.shape[1]

  # Constructing <ij|ab> and <ia|bj>
  # hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
  eri_ao = mol.intor('int2e')
  eri_ao_anti_chem = eri_ao - np.einsum("prqs->pqrs", eri_ao, optimize='optimal')
  eri_ao_phys = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
  eri_ao_anti = eri_ao - np.einsum("prqs->prsq", eri_ao_phys, optimize='optimal')

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

  e_rpa, X, Y = helper.super_matrix_solver(A_iajb, B_iajb)

  return np.sort(e_rpa)



def RPA_spatial(mol,myhf,n_occ) -> np.ndarray:
  
  mo = myhf.mo_coeff

  #print(n_occ)
  
  # Core and Virtual Orbitals (spatial orbital basis)
  o = mo[:, :n_occ]
  v = mo[:, n_occ:]

  nc = o.shape[1]
  nv = v.shape[1]

  # Constructing <ij|ab> and <ia|bj>
  # hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
  eri_ao = mol.intor('int2e')
  eri_ao_anti_chem = eri_ao - np.einsum("prqs->pqrs", eri_ao, optimize='optimal')
  eri_ao_phys = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
  eri_ao_anti = eri_ao - np.einsum("prqs->prsq", eri_ao_phys, optimize='optimal')

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
  e_rpa, X, Y = helper.super_matrix_solver(A_iajb, B_iajb)
  t2 = Y@np.linalg.inv(X)
  t2 = t2.reshape(nc,nv,nc,nv)
  t2 = t2.transpose(0,2,1,3)

  goovv,_,_,govvo,govov = helper.build_double_ints(o,v,eri_ao_phys)
  
  hsing = np.zeros((nc,nv,nc,nv))
  htrip = np.zeros((nc,nv,nc,nv))
  
  fock = np.einsum("ai,ab,ij->iajb",de,np.identity(nv),np.identity(nc),optimize="optimal")

  iabj = 2*np.einsum("iabj->iajb", govvo, optimize='optimal')
  iajb = - govov
  ikbc_ikcb = 2*goovv - np.einsum("ikcb->ikbc", goovv, optimize='optimal')
  term3 = np.einsum("ikbc,jkca->iajb", ikbc_ikcb, t2, optimize='optimal')

  hsing = fock + iabj +iajb + term3
  htrip = fock + iajb - np.einsum("ikcb,jkca->iajb", goovv, t2, optimize='optimal')

  SingEner, v = np.linalg.eig(hsing.reshape((nc*nv, nc*nv)))
  TripEner, v = np.linalg.eig(htrip.reshape((nc*nv, nc*nv)))

  return np.sort(np.real_if_close(SingEner)), np.sort(np.real_if_close(TripEner)) # in eV