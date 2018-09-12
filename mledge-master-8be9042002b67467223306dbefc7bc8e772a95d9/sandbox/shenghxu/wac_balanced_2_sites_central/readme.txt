To run this experiment:
1.Run ‘generate_img_from_GAN0.py’ and ‘generate_img_from_GAN1.py’. Make sure the models of  GAN0 and GAN1 are saved in the corresponding directories. These 2 files will generate batched of images and save them as ‘.npy’ files.
2.Run ‘Golden0_generate_probs.py’ and ‘Golden1_generate_probs.py’ to save the argmax probabilities.
3. Run ‘generate_label_plus.py’ to obtain the sum of golden 1 and golden 0 probabilities. You can also try to do ‘AND’ operation and only use the images with same labels given by the 2 golden classifiers. The overall result would be very similar in terms of classification accuracy.
4. Run ‘cifar10_train_site0_gold01_plus_label.py’ to use only GAN0 to train classifier or run ‘cifar10_train_site0_1_goldlabel_mix.py’ to use both GAN0 and GAN1 to train the classifier. 
