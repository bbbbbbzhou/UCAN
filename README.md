# Synthesizing Multi-Tracer PET Images for Alzheimer's Disease Patients using a 3D Unified Anatomy-aware Cyclic Adversarial Network

Bo Zhou, Rui Wang, Ming-Kai Chen, Adam P. Mecca, Ryan S. O'Dell, Christopher H. Van Dyck, Richard E. Carson, James S. Duncan, Chi Liu

International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2021

[[Paper](https://www.xxx)]

This repository contains the PyTorch implementation of UCAN.

### Citation
If you use this code for your research or project, please cite:

    @inproceedings{zhou2021anatomy,
      title={Synthesizing multi-tracer PET images for Alzheimer's disease patients using a 3D unified anatomy-aware cyclic adversarial network},
      author={Zhou, Bo and Wang, Rui and Chen, Ming-Kai and Mecca, Adam P. and O'Dell, Ryan S. and Van Dyck, Christopher H. and Carson, Richard E. and Duncan, James S and Liu, Chi},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={xxx},
      year={2021},
      organization={Springer}
    }


### Environment and Dependencies
Requirements:
* Python 3.6
* Pytorch 1.4.0
* scipy
* scikit-image
* nibabel
* pillow

Our code has been tested with Python 3.7, Pytorch 1.4.0, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    ./data/                          # data setup for both training and test
    ├── train_flist.npy            # file list for training cases
    ├── valid_flist.npy            # file list for validation/test cases
    │
    ├── case1A.mat                 # case 1's tracer A imaging data
    ├── case1B.mat                 # case 1's tracer B imaging data
    ├── case1C.mat                 # case 1's tracer C imaging data
    ├── case1MR.mat                # case 1's MRI imaging data
    ├── ...
    ├── case28A.mat   
    ├── case28B.mat 
    ├── case28C.mat 
    ├── case28MR.mat  
    ├── ...
    ├── case35A.mat   
    ├── case35B.mat 
    ├── case35C.mat
    ├── case35MR.mat       
    └── 

train_flist.npy contains an array with the shape of (n_train,), and the training case names are stored in it. For example,

    case1 
    case2
    case3 
    ...
    case28    

valid_flist.npy contains an array with the shape of (n_test,), and the test case names are stored in it. For example,

    case29
    ...
    case35  

Examples of train_flist.npy and valid_flist.npy are provided in './data/'. The case name stored in .npy will index the .mat files stored in './data/', where the .mat files contain imaging data. For example, case1A.mat contains case 1's PET tracer A imaging data, while case1B.mat and case1C.mat contain case 1's tracer B and tracer C imaging data, respectively.


### To Run Our Code
- Train the model
```bash
python main.py \
--name 'experiment_PET3_DUSENET' \
--mode 'train' \
--g 'DUSENET' \
--g_repeat_num 3 \
--d 'DC' \
--d_repeat_num 4 \
--train_file './data/train_flist.npy' \
--test_file './data/valid_flist.npy' \
--dataset 'PET3' \
--dataroot './data' \
--n_patch_train 100 \
--patch_size_train 64 64 64 \
--n_patch_test 1 \
--patch_size_test 80 96 80 \
--batch_size 16 \
--aug \
--use_MR
```
where \
`--use_MR` defines whether to use MR information. \
`--aug` defines whether to use data augumentation (random flipping) during training. \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python main.py \
--name 'experiment_PET3_DUSENET' \
--mode 'test' \
--g 'DUSENET' \
--g_repeat_num 3 \
--d 'DC' \
--d_repeat_num 4 \
--train_file './data/train_flist.npy' \
--test_file './data/valid_flist.npy' \
--test_iters 50000 \
--dataset 'PET3' \
--dataroot './data' \
--n_patch_test 1 \
--patch_size_test 80 96 80 \
--use_MR
```
where \
`--test_iters` defines the training iteration to use in the validation. \
Sample train/test scripts are provided under './scripts/' and can be directly executed.


### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```