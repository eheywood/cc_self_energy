from pyscf import gto, scf, cc
from pyscf.cc.bccd import bccd_kernel_
from pyscf import ao2mo

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

# 2. Do a standard CCSD
mycc = cc.CCSD(mf).run()

# 3. Brückner-orbital optimization
#    - conv_tol_normu: tolerance on orbital rotation
#    - max_cycle: how many outer BCCD cycles to do
#    - canonicalization: semi-canonicalize BOs at each step
mycc = bccd_kernel_(mycc,
                    conv_tol_normu=1e-6,
                    max_cycle=30,
                    diis=True,
                    canonicalization=True,
                    verbose=4)

# After this:
#  - mycc.t1 amplitudes ≃ 0
#  - mycc._scf.mo_coeff are now Brückner orbitals
print("Brückner CCSD energy:", mycc.e_tot)