from pyscf import gto, scf

mol = gto.Mole()
mol.atom = '''
H 0 0 0
H 0 0 0.74
'''
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

