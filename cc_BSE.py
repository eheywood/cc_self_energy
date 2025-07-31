import numpy as np
from pyscf import gto, scf, cc
from scipy.linalg import block_diag
from BSE_Helper import spinor_one_and_two_e_int


np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493


def bccd_t2_amps(mol:gto.Mole) -> tuple[np.ndarray,np.ndarray]:
    myhf = mol.HF.run() 
    mycc = cc.BCCD(myhf).run()
    mo = mycc.mo_coeff

    print(f'Max. value in BCCD T1 amplitudes {abs(mycc.t1).max()}')
    print(f'Max. value in BCCD T2 amplitudes {abs(mycc.t2).max()}')
    # print(mycc.t2.shape)


    return mo, mycc.t2
    
def get_spinorbs(mo:np.ndarray) -> tuple[np.ndarray,np.ndarray]:

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

    return core_spinorbs, vir_spinorbs

def get_self_energy(t2:np.ndarray, oovv:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    occ_selfeng = np.einsum("ikab,jkab -> ij", oovv,t2,optimize="optimal")
    vir_selfeng = np.einsum("ijbc,ijac -> ab", oovv,t2,optimize="optimal")

    return occ_selfeng,vir_selfeng


if __name__ == "__main__":

    # Define Molecule to calculate amplitudes and mo for
    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
                basis='cc-pVTZ',
                spin=0,
                symmetry=False,
                unit="Bohr")
    
    # get molecular orbitals and t2 amplitudes
    mo,t2 = bccd_t2_amps(mol)
    t2_spin = np.block([[t2,t2],
                        [t2,t2]])
    print(t2_spin.shape)
    core_spinorbs, vir_spinorbs = get_spinorbs(mo)

    # Build electron repulsion integrals
    _, eri_ao = spinor_one_and_two_e_int(mol)                       # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # Build the required anti-symmetrised orbitals
    oovv = np.einsum("pi,qj,pqrs,ra,sb->ijab", core_spinorbs, core_spinorbs, anti_eri_ao, vir_spinorbs, vir_spinorbs, optimize="optimal")
    ooov =  np.einsum("pi,qk,pqrs,rj,sc->ikjc",core_spinorbs,core_spinorbs,anti_eri_ao,core_spinorbs,vir_spinorbs,optimize="optimal") 
    vovv =  np.einsum("pa,qk,pqrs,rb,sc->akbc",vir_spinorbs,core_spinorbs,anti_eri_ao,vir_spinorbs,vir_spinorbs,optimize="optimal") 
    ovvo = np.einsum("pi,qa,pqrs,rb,sj->iabj", core_spinorbs, vir_spinorbs, anti_eri_ao, vir_spinorbs, core_spinorbs,optimize="optimal") 
    ovov = np.einsum("pi,qa,pqrs,rj,sb->iajb", core_spinorbs, vir_spinorbs, anti_eri_ao, core_spinorbs, vir_spinorbs, optimize="optimal")

    occ_selfeng, vir_selfeng = get_self_energy(t2,oovv)
    print(occ_selfeng)


