import numpy as np
from pyscf import gto, scf, cc
from scipy.linalg import block_diag
from BSE_Helper import spinor_one_and_two_e_int

np.set_printoptions(precision=6, suppress=True, linewidth=100000)
eV_to_Hartree = 0.0367493


def bccd_t2_amps(mol:gto.Mole) -> tuple[np.ndarray,np.ndarray]:
    myhf = mol.HF.run() 
    mycc = cc.BCCD(myhf,conv_tol_normu=1e-8).run()

    print(mycc.e_tot)
    mo = mycc.mo_coeff

    print(f'Max. value in BCCD T1 amplitudes {abs(mycc.t1).max()}')
    print(f'Max. value in BCCD T2 amplitudes {abs(mycc.t2).max()}')

    t2 = mycc.t2

    # Get number of spatial orbitals
    n_occ = t2.shape[0]
    n_vir = t2.shape[2]

    t_ijab = t2
    print(t2.reshape(-1))
    t_ijba = -np.einsum("ijab->ijba", t2,optimize='optimal')

    t2_spin = np.zeros((n_occ*2,n_occ*2,n_vir*2,n_vir*2))
    t2_spin[:n_occ,:n_occ,:n_vir,:n_vir] = t_ijab-t_ijba
    t2_spin[n_occ:,n_occ:,n_vir:,n_vir:] = t_ijab-t_ijba
    t2_spin[:n_occ,n_occ:,:n_vir,n_vir:] = t_ijab
    t2_spin[n_occ:,:n_occ,n_vir:,:n_vir] = t_ijab
    t2_spin[n_occ:,:n_occ,:n_vir,n_vir:] = t_ijba
    t2_spin[:n_occ,n_occ:,n_vir:,:n_vir] = t_ijba

    # print(t2_spin)
    return mo, t2_spin
    
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
    """_summary_

    Parameters
    ----------
    t2 : np.ndarray
        t2 amplitudes
    oovv : np.ndarray
        anti symmetrised integral

    Returns
    -------
    tuple[np.ndarray,np.ndarray]
        occupied self energy, virtual self energy 
    """
    occ_selfeng = 0.5 * np.einsum("ikab,jkab -> ij", oovv,t2,optimize="optimal")
    vir_selfeng = -0.5 * np.einsum("ijbc,ijac -> ab", oovv,t2,optimize="optimal")

    return occ_selfeng, vir_selfeng

def build_fock_matrices(mol)-> tuple[np.ndarray,np.ndarray]:

    mf = scf.HF(mol)     
    mf.kernel()          
    F_ao = mf.get_fock()    
    C   = mf.mo_coeff        
    F_mo = C.T @ F_ao @ C   
    fock_occ = F_mo[:int(n_occ/2),:int(n_occ/2)]
    fock_vir = F_mo[int(n_occ/2):,int(n_occ/2):]

    #TODO: COME BACK AND GENERALISE TO LARGER SYSTEMS
    spat_occ = int(n_occ/2)
    spat_vir = int(n_vir/2)
    
    fock_occ_spin = np.zeros((n_occ,n_occ))
    fock_vir_spin = np.zeros((n_vir,n_vir))

    fock_occ_spin[:spat_occ,:spat_occ] = fock_occ
    fock_occ_spin[spat_occ:,spat_occ:] = fock_occ

    fock_vir_spin[:spat_vir,:spat_vir] = fock_vir
    fock_vir_spin[spat_vir:,spat_vir:] = fock_vir

    return fock_occ_spin, fock_vir_spin

if __name__ == "__main__":

    # Define Molecule to calculate amplitudes and mo for
    mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
                basis='cc-pVTZ',
                spin=0,
                symmetry=False,
                unit="Bohr")
    
    # get molecular orbitals and t2 amplitudes
    mo,t2 = bccd_t2_amps(mol)   
    core_spinorbs, vir_spinorbs = get_spinorbs(mo)
    n_occ = core_spinorbs.shape[1]
    n_vir = vir_spinorbs.shape[1]

    print(t2.shape)
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

    # n_occ x n_occ, n_vir x n_vir
    occ_selfeng, vir_selfeng = get_self_energy(t2,oovv)

    # n_occ x n_occ, n_vir x n_vir
    fock_occ, fock_vir = build_fock_matrices(mol)

    F_ij = occ_selfeng + fock_occ
    F_ab = vir_selfeng + fock_vir    

    F_abij = np.einsum('ab, ij -> iajb', F_ab, np.identity(n_occ),optimize='optimal')
    F_ijab = np.einsum('ij, ab -> iajb', F_ij, np.identity(n_vir),optimize='optimal')
    
    H_bse = F_abij - F_ijab + np.einsum("iabj->iajb", ovvo,optimize='optimal') + np.einsum("ikbc,jkca -> iajb", oovv,t2,optimize="optimal")

    print(H_bse)

