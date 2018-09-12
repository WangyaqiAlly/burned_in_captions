import numpy as np
import matplotlib.pyplot as plt

txtfile = 'ctgan-semi-000-of-010-sites.txt'
pngfile = txtfile[:-3]+'png'

mat = np.loadtxt(txtfile)

plt.figure()
# plt.plot(mat[:,0], mat[:,1])
# plt.subplot(211)
# plt.plot(mat[:,0])
# plt.ylabel('Generator Cost')
# plt.subplot(212)
plt.plot(-mat[:,1])
plt.plot(mat[:,2])
plt.ylabel('Neg. Discriminator Cost/W Distance')
plt.xlabel('Generator Iteration')
# plt.xlim(20000, 50000)
plt.savefig(pngfile)
