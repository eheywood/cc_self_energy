from pyscf import gto, scf, tdscf

hartree_to_eV = 27.211386245988

mol = gto.M(
    atom = """C -0.00234503 0.00000000 0.87125063
              C -1.75847785 0.00000000 -1.34973671
              O  2.27947397 0.00000000 0.71968028
              H -0.92904537 0.00000000 2.73929404
              H -2.97955463 1.66046488 -1.25209463
              H -2.97955463 -1.66046488 -1.25209463
              H -0.70043433 0.00000000 -3.11066412""",
    basis = "aug-cc-pVTZ",  
    spin = 0,
    symmetry = False,
    unit = "Bohr",
)

mf = scf.RHF(mol).density_fit()  # density fitting speeds things up
mf.conv_tol = 1e-9
mf.run()

# TDHF (RPA) singlets
td_s = tdscf.TDHF(mf)
td_s.kernel(nstates=10)

print("\nSinglet excitations")
for i, e in enumerate(td_s.e, 1):
    print(f"Singlet {i:2d}: {e*hartree_to_eV:8.3f} eV")

# TDHF (RPA) triplets
td_t = tdscf.TDHF(mf).set(singlet=False)
td_t.kernel(nstates=nroots)

print("\n== TDHF (RPA) Triplet excitations ==")
for i, e in enumerate(td_t.e, 1):
    print(f"Triplet {i:2d}: {e*hartree_to_eV:8.3f} eV")

# Optional: quick analysis of leading configurations
# td_s.analyze()

# # Mean-field (DF speeds up AO→aux transforms that RPA needs)
# myhf = scf.RHF(mol).density_fit().run()
# td_s = tdscf.TDHF(myhf)
# td_s.kernel(nstates=10) 