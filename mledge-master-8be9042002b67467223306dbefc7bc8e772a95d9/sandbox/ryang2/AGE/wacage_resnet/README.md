## To run WAC-AGE
To directly run, download the whole directory, create ./data/ folder and put the cifar-10 data batch directly under the ./data/ folder. Then run
```
CUDA_VISIBLE_DEVICES='1' python <filename>
```

Or it needs to change the data directory inside the code.

## Data generation and classification
The batch size here are all 128 due to the golden classifier requirement.

#### wacage_data_generation program:
Needs to define model path, and save path.

*Requirement*: tflib folder, trained model

#### wacage_recon program:
Reconstruct all the real data four times. The user could reduce the iteration number to reduce replication.

*Requirement*: tflib folder, CIFAR-10 data (DATA_DIR in code), trained model

As for other related generation, such as z-space for real image, please go to hydra-04.cisco.com:/home/guest/ryang2/AGE/cifar10/wage_resnet/data_generation