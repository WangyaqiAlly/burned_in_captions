import numpy as np
import matplotlib.pyplot as plt

disc_prob = np.load('wacage_labelprob_record/disc_prob.npy')
real_disc_prob = np.load('wacage_labelprob_record/real_disc_prob.npy')
golden_prob = np.load('wacage_labelprob_record/golden_prob.npy')
real_golden_prob = np.load('wacage_labelprob_record/real_gold_prob.npy')

# print disc_prob.shape
# print disc_prob[1:100]
plt.figure(1) 
plt.hist(disc_prob, bins='auto')    
plt.title("Histogram, Discriminator")

plt.figure(2)
plt.hist(golden_prob, bins='auto')
plt.title("Histogram, golden classifier")

plt.figure(3)
plt.hist(real_disc_prob, bins='fd')
plt.xlim((-0.1, 1.1))
plt.title("Histogram, Discriminator, real image")

plt.figure(4)
plt.hist(real_golden_prob, bins='fd')
plt.xlim((-0.1, 1.1))
plt.title("Histogram, golden classifier, real image")
plt.show()