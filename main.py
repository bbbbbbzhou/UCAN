import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    config.log_dir = os.path.join('outputs', config.name, config.log_dir)
    config.model_save_dir = os.path.join('outputs', config.name, config.model_save_dir)
    config.sample_dir = os.path.join('outputs', config.name, config.sample_dir)
    config.result_dir = os.path.join('outputs', config.name, config.result_dir)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    if config.dataset in ['PET3']:
        pet3_train_loader = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'train', config.train_file,
                                       None, None, 
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       config.batch_size, config.num_workers)
        pet3_test_loader1 = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'valid', config.test_file,
                                       0, 1,
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       1, config.num_workers)
        pet3_test_loader2 = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'valid', config.test_file,
                                       0, 2,
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       1, config.num_workers)
        pet3_test_loader3 = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'valid', config.test_file,
                                       1, 0,
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       1, config.num_workers)
        pet3_test_loader4 = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'valid', config.test_file,
                                       1, 2,
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       1, config.num_workers)
        pet3_test_loader5 = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'valid', config.test_file,
                                       2, 0,
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       1, config.num_workers)
        pet3_test_loader6 = get_loader(config.dataset, config.dataroot, config.c_dim,
                                       config.aug, 'valid', config.test_file,
                                       2, 1,
                                       config.patch_size_train, config.n_patch_train, config.patch_size_test, config.n_patch_test,
                                       config.norm_A, config.norm_B, config.norm_C, config.norm_MR,
                                       1, config.num_workers)

    # Solver for training and testing.
    solver = Solver(pet3_train_loader, 
                    pet3_test_loader1, pet3_test_loader2, 
                    pet3_test_loader3, pet3_test_loader4, 
                    pet3_test_loader5, pet3_test_loader6,
                    config)

    if config.mode == 'train':
        if config.dataset in ['PET3']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['PET3']:
            solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='xxx', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=3, help='dimension of domain labels')

    parser.add_argument('--g', type=str, default='DUSENET', help='DUSENET / UNET / RAE (Residual AutoEncoder)')
    parser.add_argument('--d', type=str, default='DC', help='D:classify R or F / C:classify domain')

    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=3, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=4, help='number of strided conv layers in D')

    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=5, help='weight for reconstruction loss')
    parser.add_argument('--lambda_pair', type=float, default=15, help='weight for pair-wise reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    parser.add_argument('--use_MR', default=False, action='store_true', help='use MR as input prior')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='PET3', choices=['PET3', 'XXX'])
    parser.add_argument('--train_file', type=str,
                        default='./../DATA/Processed/PET_3Tracer/processed_3D_All_5fold/train_flist.npy',
                        help='train file list - fold x')
    parser.add_argument('--dataroot', type=str, default='./../DATA/Processed/PET_3Tracer/processed_3D_All_5fold', help='data root')
    parser.add_argument('--norm_A', type=float, default=1, help='A normalization by dividing')
    parser.add_argument('--norm_B', type=float, default=1, help='B normalization by dividing')
    parser.add_argument('--norm_C', type=float, default=1, help='C normalization by dividing')
    parser.add_argument('--norm_MR', type=float, default=1, help='MR normalization by dividing')

    parser.add_argument('--n_patch_train', type=int, default=8, help='# of patch cropped for each image')
    parser.add_argument('--patch_size_train', nargs='+', type=int, default=[64, 64, 64], help='patch size to crop')
    parser.add_argument('--aug', default=False, action='store_true', help='use augmentation')
    parser.add_argument('--batch_size', type=int, default=3, help='mini-batch size')

    parser.add_argument('--num_iters', type=int, default=2000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=3, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--n_patch_test', type=int, default=1, help='# of patch cropped for each image')
    parser.add_argument('--patch_size_test', nargs='+', type=int, default=[80, 96, 80], help='patch size to crop')
    parser.add_argument('--test_file', type=str,
                        default='./../DATA/Processed/PET_3Tracer/processed_3D_All_5fold/valid_flist.npy',
                        help='test file list - fold x')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--validate_step', type=int, default=500)

    config = parser.parse_args()
    print(config)
    main(config)
