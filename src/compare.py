import numpy as np

spatial = np.loadtxt("spatial_eig.csv",dtype='float',delimiter=',')
spin = np.loadtxt("spin_eig.csv",dtype='float',delimiter=',')

for i in range(spatial.shape[0]):
  isTrue = False
  for j in range(spin.shape[0]):
    diff = np.abs(spin[j] - spatial[i])
    if diff < 1e-4:
      isTrue = isTrue | True
      continue
  
  print(isTrue)
