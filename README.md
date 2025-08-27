# Excitation energies from Coupled Cluster Greens Function Theory

A number of different levels of theory have been implemented to calculate excitation energies. Following [Coveney and Tew's paper](https://arxiv.org/abs/2309.10451):
- `cc_BSE_spinorb.py`: CC-BSE in Spin-orbital form 
- `RPA.py`: RPA in Spin-Orbital form 
- `GW_BSE.py`:GW-BSE in Spin-orbital form 

Alternative spin-free forms of the following are also implemented:
- `RPA_spatial.py`: RPA using two different ways methods to extract excitation energies. Spatial orbitals.
- `cc_BSE_spatialorb.py`: CC-BSE in Spatial-orbital form. 

Use the `cc_controller.py` script to define your molecule and basis set and control which of the calculations desired to run.

## Testing and Results Output ##
Use command `pytest` to generate the results files for H2, Be, NH3, H2O and C2. Further molecules can be included in testing, by appending the respective geometries, basis set and name to the array of molecules in `tests/test_helper.py`. 

