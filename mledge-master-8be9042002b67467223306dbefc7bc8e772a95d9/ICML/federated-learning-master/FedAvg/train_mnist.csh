#/bin/csh!


# centralized
CUDA_VISIBLE_DEVICES='0' python ./main_nn.py --dataset mnist \
					     --model cnn \
					     --num_channels 1 \
					     --iid 1  \
					     --seed 123456 \
					     --lr 0.03 \
					     --epochs 100 > mnist-cent.rec 

# federated
CUDA_VISIBLE_DEVICES='0' python ./main_fed.py --dataset mnist \
					     --model cnn \
					     --num_channels 1 \
					     --iid 1  \
					     --seed 123456 \
					     --lr 0.03 \
					     --epochs 100 > mnist-fed.rec 


 
mkdir trial_lr_0.03
mv *.rec trial_lr_0.03/.
mv *.txt trial_lr_0.03/.
mv ../save/*.png trial_lr_0.03/.

