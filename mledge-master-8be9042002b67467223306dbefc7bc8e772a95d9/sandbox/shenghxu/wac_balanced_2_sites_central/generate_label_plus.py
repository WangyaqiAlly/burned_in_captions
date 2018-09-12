import numpy as np

for b_n in xrange(60000):
	l0 = np.load('./data0/gold_prob_site1'+('%06d' % b_n)+'.npy')
	l1 = np.load('./data0/gold_prob_site0'+('%06d' % b_n)+'.npy')

	l_plus = l0+l1

	label = np.argmax(l_plus,axis = 1)
	label = label.astype('uint8')

	np.save('./data0/gold_01_plus_label'+('%06d' % b_n), label)

	if b_n % 1000 ==0:
		print b_n