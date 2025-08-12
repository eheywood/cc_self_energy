from pyscf import gto, scf, tdscf, gw

hartree_to_eV = 27.211386245988

mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
        basis='cc-pVTZ',
        spin=0,
        symmetry=False,
        unit="Bohr")

# mol = gto.M(
#     atom = """C -0.00234503 0.00000000 0.87125063
#               C -1.75847785 0.00000000 -1.34973671
#               O  2.27947397 0.00000000 0.71968028
#               H -0.92904537 0.00000000 2.73929404
#               H -2.97955463 1.66046488 -1.25209463
#               H -2.97955463 -1.66046488 -1.25209463
#               H -0.70043433 0.00000000 -3.11066412""",
#     basis = "aug-cc-pVTZ",  
#     spin = 0,
#     symmetry = False,
#     unit = "Bohr",
# )

mf = scf.RHF(mol).density_fit()  # density fitting speeds things up
mf.conv_tol = 1e-9
mf.run()

ener = gw.rpa.RPA(mf)