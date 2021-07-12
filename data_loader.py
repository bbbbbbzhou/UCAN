from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
from os import listdir
from os.path import join, isfile
from scipy import io as sio
from PIL import Image
import torch
import os
import random
from util.data_patch_util import *


def get_loader(dataset, dataroot, c_dim,
               aug, phase, phase_file,
               c_org, c_trg,
               patch_size_train, n_patch_train, patch_size_test, n_patch_test,
               norm_A, norm_B, norm_C, norm_MR,
               batch_size, num_workers):

    if dataset == 'PET3':
        dataset = PET3(dataroot, c_dim,
                       aug, phase, phase_file,
                       c_org, c_trg,
                       patch_size_train, n_patch_train, patch_size_test, n_patch_test,
                       norm_A, norm_B, norm_C, norm_MR)
    elif dataset == 'XXX':
        dataset = None

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(phase == 'train'),
                                  num_workers=num_workers)
    return data_loader


class PET3(data.Dataset):
    def __init__(self, dataroot, c_dim,
                 aug, phase, phase_file,
                 c_org, c_trg,
                 patch_size_train, n_patch_train, patch_size_test, n_patch_test,
                 norm_A, norm_B, norm_C, norm_MR):
        self.dataroot = dataroot
        self.c_dim = c_dim
        self.phase = phase
        self.phase_file = phase_file
        self.aug = aug

        self.c_org = c_org
        self.c_trg = c_trg

        self.flist = np.load(self.phase_file)
        self.num_images = len(self.flist)
        print('Number of original images: {}'.format(self.num_images))

        # load all images and patching
        if self.phase == 'train':
            self.patch_size = patch_size_train
            self.n_patch = n_patch_train
        elif self.phase == 'valid':
            self.patch_size = patch_size_test
            self.n_patch = n_patch_test

        self.A_all = []
        self.B_all = []
        self.C_all = []
        self.MR_all = []

        for f in self.flist:
            print('Patching: ' + str(f))

            # create the random index for cropping patches
            X_template = self.read_mat(join(self.dataroot, f + 'A.mat'), var_name="img")
            indexes = get_random_patch_indexes(data=X_template,
                                               patch_size=self.patch_size, num_patches=self.n_patch,
                                               padding='VALID')

            # use index to crop patches
            X = self.read_mat(join(self.dataroot, f + 'A.mat'), var_name="img")
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.A_all.append(X_patches)

            X = self.read_mat(join(self.dataroot, f + 'B.mat'), var_name="img")
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.B_all.append(X_patches)

            X = self.read_mat(join(self.dataroot, f + 'C.mat'), var_name="img")
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.C_all.append(X_patches)

            X = self.read_mat(join(self.dataroot, f + 'MR.mat'), var_name="img")
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.MR_all.append(X_patches)

        self.A_all = (np.concatenate(self.A_all, 0) / norm_A - 0.5) * 2
        self.B_all = (np.concatenate(self.B_all, 0) / norm_B - 0.5) * 2
        self.C_all = (np.concatenate(self.C_all, 0) / norm_C - 0.5) * 2
        self.MR_all = (np.concatenate(self.MR_all, 0) / norm_MR - 0.5) * 2

        # calculate the final total number after augmentation
        self.l = len(self.A_all)

    def __getitem__(self, index):
        A_tmp = self.A_all[index]
        B_tmp = self.B_all[index]
        C_tmp = self.C_all[index]
        MR_tmp = self.MR_all[index]

        # data augmentation for training
        if self.aug and self.phase == 'train':
            if random.randint(0, 1):
                A_tmp = np.flip(A_tmp, axis=1)
                B_tmp = np.flip(B_tmp, axis=1)
                C_tmp = np.flip(C_tmp, axis=1)
                MR_tmp = np.flip(MR_tmp, axis=1)

            if random.randint(0, 1):
                A_tmp = np.flip(A_tmp, axis=2)
                B_tmp = np.flip(B_tmp, axis=2)
                C_tmp = np.flip(C_tmp, axis=2)
                MR_tmp = np.flip(MR_tmp, axis=2)

            if random.randint(0, 1):
                A_tmp = np.flip(A_tmp, axis=3)
                B_tmp = np.flip(B_tmp, axis=3)
                C_tmp = np.flip(C_tmp, axis=3)
                MR_tmp = np.flip(MR_tmp, axis=3)

            if random.randint(0, 1):
                A_tmp = np.rot90(A_tmp, axes=(1, 2))
                B_tmp = np.rot90(B_tmp, axes=(1, 2))
                C_tmp = np.rot90(C_tmp, axes=(1, 2))
                MR_tmp = np.rot90(MR_tmp, axes=(1, 2))

            if random.randint(0, 1):
                A_tmp = np.rot90(A_tmp, axes=(1, 3))
                B_tmp = np.rot90(B_tmp, axes=(1, 3))
                C_tmp = np.rot90(C_tmp, axes=(1, 3))
                MR_tmp = np.rot90(MR_tmp, axes=(1, 3))

            if random.randint(0, 1):
                A_tmp = np.rot90(A_tmp, axes=(2, 3))
                B_tmp = np.rot90(B_tmp, axes=(2, 3))
                C_tmp = np.rot90(C_tmp, axes=(2, 3))
                MR_tmp = np.rot90(MR_tmp, axes=(2, 3))

        # label for training (random org and trg) and test (fix org and trg)
        if self.phase == 'train':
            c_org = torch.randint(low=0, high=self.c_dim, size=(1,))[0]
            c_trg = torch.randint(low=0, high=self.c_dim, size=(1,))[0]
        elif self.phase == 'valid':
            c_org = self.c_org
            c_trg = self.c_trg

        if c_org == 0:
            x_org = torch.FloatTensor(A_tmp.copy())
        if c_org == 1:
            x_org = torch.FloatTensor(B_tmp.copy())
        if c_org == 2:
            x_org = torch.FloatTensor(C_tmp.copy())

        if c_trg == 0:
            x_trg = torch.FloatTensor(A_tmp.copy())
        if c_trg == 1:
            x_trg = torch.FloatTensor(B_tmp.copy())
        if c_trg == 2:
            x_trg = torch.FloatTensor(C_tmp.copy())

        x_MR = torch.FloatTensor(MR_tmp.copy())

        return x_org, x_trg, c_org, c_trg, x_MR

    def __len__(self):
        """Return the number of images after augmentation."""
        return self.l

    @staticmethod
    def read_mat(filename, var_name="img"):
        mat = sio.loadmat(filename)
        return mat[var_name]

