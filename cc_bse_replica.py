from pyscf import gto, scf, cc
from pyscf.cc.bccd import bccd_kernel_
from pyscf import ao2mo
import pyscf
import numpy as np
from pyscf.lib import misc

print(pyscf.__version__)

if __name__ == "__main__":

    # 1. Build molecule & do H2
    mol = gto.M(atom='H 0 0 0; H 0 0 2', 
                basis='cc-pvtz', 
                unit  = 'Bohr',
                charge= 0,              # $charge from dscf = 0
                spin  = 0,              # closed‐shell
                symmetry = False,       # $symmetry c1
                verbose = 4,
            )
    mf  = scf.RHF(mol).run()

    # Set up indicies for orbital counting. 
    nmo = mf.mo_coeff.shape[0]
    nelec = mol.nelec 
    
    nocc = nelec[0]

    # Virtual orbitals
    nvir = nmo - nocc

    print(f"Total MOs: {nmo}")
    print(f"Occupied: {nocc}")
    print(f"Virtual: {nvir}")

    
    hbse = np.zeros((nocc,nocc,nvir,nvir,4)) # 4 spin cases...
    # spin case 0: up up up up
    # spin case 1: down down down down
    # spin case 2: up down up down
    # spin case 3s: down up down up

    # Load integrals into memory:

    # loop over spin cases, then spatial orbitals and populate
    for s in range(4):
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        # would populate with gfock (self energy) here

                        for k in range(nocc):
                            for c in range(nvir):
                                # add to HBSE with the relative integrals

                                # how to index using spin cases
                                hbse += 1
