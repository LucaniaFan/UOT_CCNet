import numpy as np

file = np.load('./results/15_SENSE_cnt.py.npy')
print(file) 
np.savetxt('./results/15_SENSE_cnt.txt', file)
