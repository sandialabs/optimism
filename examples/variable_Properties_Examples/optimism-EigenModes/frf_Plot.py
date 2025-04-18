import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.optimize import curve_fit
# from scipy import interpolate
# from scipy import signal
# file = '~/projects/dirrect_numerical_simulation_of_foam_replacement_structures/runs/lucas_folder/preload_example/Edited/workdir.1/preload/heartbeat/'

frf = pd.read_csv('frfTest.txt')
print(frf.columns)

# frf_f = [freq[0] for freq in frf.columns[0]]
# frf_r = [resp[0] for resp in frf.columns[1]]

# print(frf_f)
# print(frf_r)

# plt.plot(frf_f,frf_r)
# plt.savefig('test1.png', dpi=600)
# plt.close()