# Training MoCoV2 on CIFAR10 Dataset

Here are some notebooks and scripts to train MoCoV2 on CIFAR10 (and easily adjustible for other datasets).

Soon I will make this readme better with more info on how to tune this model yourself, and add scripts so you dont have to do it through a jupyter notebook.

# Instructions
- Download CIFAR10 in folders from Kaggle: https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders

# Results
After 3000 epochs (resnet18):

<p float="left">
  <img src="./imgs/results/3000eps/train_loss_ssl.png?raw=true" width="400" />
  <br>
  <img src="./imgs/results/3000eps/downstream_max_val_acc_8813.png?raw=true" width="300" /> 
  <img src="./imgs/results/3000eps/downstream_min_trian_loss.png?raw=true" width="300" />
</p>

- DOWNSTREAM_max_val_acc: every 50 epochs, training was paused, the model was frozen, and the `fc` was replaced with 2 linear layers. Only these 2 layers were trained for 60 epochs to acheve a final "downstream task" accuracy of 88.13% in only 60 epochs on test data.
```
# MODEL hyperparams
memory_bank_size = 4096
moco_max_epochs = 3000
downstream_max_epochs = 60
downstream_test_every = 50
# DATA hyperparams
moco_batch_size = 512
classifier_train_batch_size = 512
classifier_test_batch_size = 512
```
- This took about 2.5 days with an RTX2070

# Notes
- MoCoV2 implementation was made by taking the MoCoV1 implementation from [lightly](https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html) and making some tweaks.
- Original MoCo paper: https://arxiv.org/abs/1911.05722
- MoCoV2 paper: https://arxiv.org/abs/2003.04297
- CIFAR10 website: https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/

### TODO
- figure out why histograms dont work in moco_model.py
- Describe files/classes/functions
