from model import Generator_DUSENET
from model import Discriminator_DC
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import nibabel as nib
from util.util import *
import numpy as np
import os
import time
import datetime
from tqdm import tqdm
import csv


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, pet3_train_loader,
                 pet3_test_loader1, pet3_test_loader2, 
                 pet3_test_loader3, pet3_test_loader4,
                 pet3_test_loader5, pet3_test_loader6,
                 config):
        """Initialize configurations."""

        self.config = config

        # Data loader.
        self.pet3_train_loader = pet3_train_loader
        self.pet3_test_loader1 = pet3_test_loader1
        self.pet3_test_loader2 = pet3_test_loader2
        self.pet3_test_loader3 = pet3_test_loader3
        self.pet3_test_loader4 = pet3_test_loader4
        self.pet3_test_loader5 = pet3_test_loader5
        self.pet3_test_loader6 = pet3_test_loader6

        # Model configurations.
        self.c_dim = config.c_dim

        self.patch_size = config.patch_size_train if config.mode == 'train' else config.patch_size_test

        self.g = config.g
        self.d = config.d

        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_pair = config.lambda_pair
        self.lambda_gp = config.lambda_gp

        self.use_MR = config.use_MR

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.validate_step = config.validate_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['PET3']:
            input_dim = 1
            if self.use_MR:
                input_dim = input_dim + 1

            if self.g == 'DUSENET':
                self.G = Generator_DUSENET(input_dim, self.g_conv_dim, self.c_dim, self.g_repeat_num)

            if self.d == 'DC':
                self.D = Discriminator_DC(self.patch_size[0], self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        # D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=3):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'PET3':
            data_train_loader = self.pet3_train_loader
            data_test_loader1 = self.pet3_test_loader1
            data_test_loader2 = self.pet3_test_loader2
            data_test_loader3 = self.pet3_test_loader3
            data_test_loader4 = self.pet3_test_loader4
            data_test_loader5 = self.pet3_test_loader5
            data_test_loader6 = self.pet3_test_loader6

        # Fetch fixed inputs for debugging.
        data_train_iter = iter(data_train_loader)
        x_org, x_trg, c_org, c_trg, x_MR = next(data_train_iter)

        x_fixed_org = x_org
        x_fixed_org = x_fixed_org.to(self.device)

        x_fixed_MR = x_MR
        x_fixed_MR = x_fixed_MR.to(self.device)

        c_fixed_list = self.create_labels(c_org, self.c_dim)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            data_train_iter = iter(data_train_loader)
            x_real_org, x_real_trg, label_org, label_trg, x_MR = next(data_train_iter)

            # Generate original and target domain labels randomly.
            if self.dataset == 'PET3':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real_org = x_real_org.to(self.device)           # Input images.
            x_real_trg = x_real_trg.to(self.device)
            x_MR = x_MR.to(self.device)
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing domain classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing domain classification loss.

            if self.d == 'DC':
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                # Compute loss with real images.
                out_src, out_cls = self.D(x_real_org)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org)

                # Compute loss with fake images.
                inp = x_real_org
                if self.use_MR:
                    inp = torch.cat([inp, x_MR], 1)
                x_fake_trg = self.G(inp, c_trg)
                out_src, out_cls = self.D(x_fake_trg.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real_org.size(0), 1, 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real_org.data + (1 - alpha) * x_fake_trg.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    inp = x_real_org
                    if self.use_MR:
                        inp = torch.cat([inp, x_MR], 1)

                    x_fake_trg = self.G(inp, c_trg)
                    out_src, out_cls = self.D(x_fake_trg)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg)

                    # Target-to-original domain.
                    inp = x_fake_trg
                    if self.use_MR:
                        inp = torch.cat([inp, x_MR], 1)

                    x_reconst = self.G(inp, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real_org - x_reconst))

                    # Target-target paired loss
                    g_loss_pair = torch.mean(torch.abs(x_fake_trg - x_real_trg))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_pair * g_loss_pair
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                    loss['G/loss_pair'] = g_loss_pair.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed_org]
                    for c_fixed in c_fixed_list:
                        inp = x_fixed_org
                        if self.use_MR:
                            inp = torch.cat([inp, x_fixed_MR], 1)
                        x_fake_list.append(self.G(inp, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=4)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    ss = x_concat.shape
                    save_image(self.denorm(x_concat.data.cpu().view(ss[0], ss[1], ss[2], -1)), sample_path, nrow=1, padding=1)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # =================================================================================== #
            #                                 5. Validation on test set                           #
            # =================================================================================== #
            if (i+1) % self.validate_step == 0:
                with torch.no_grad():
                    # validate 0-1 / 0-2 / 1-0 / 1-2 / 2-0 / 2-1
                    psnr_mean_set = np.zeros(6)
                    mse_mean_set = np.zeros(6)
                    for ii in range(0, 6):
                        if ii == 0:
                            val_bar = tqdm(data_test_loader1)
                        if ii == 1:
                            val_bar = tqdm(data_test_loader2)
                        if ii == 2:
                            val_bar = tqdm(data_test_loader3)
                        if ii == 3:
                            val_bar = tqdm(data_test_loader4)
                        if ii == 4:
                            val_bar = tqdm(data_test_loader5)
                        if ii == 5:
                            val_bar = tqdm(data_test_loader6)

                        avg_psnr = AverageMeter()
                        avg_mse = AverageMeter()

                        for (x_real_org, x_real_trg, label_org, label_trg, x_MR) in val_bar:
                            # Prepare input images and target domain labels.
                            x_real_org = x_real_org.to(self.device)
                            x_MR = x_MR.to(self.device)

                            c_org = self.label2onehot(label_org, self.c_dim)
                            c_trg = self.label2onehot(label_trg, self.c_dim)
                            c_org = c_org.to(self.device)
                            c_trg = c_trg.to(self.device) 

                            # Translate images.
                            inp = x_real_org
                            if self.use_MR:
                                inp = torch.cat([inp, x_MR], 1)
                            x_fake_trg = self.G(inp, c_trg)

                            # Calculate metrics
                            psnr_ = psnr(x_fake_trg.cpu(), x_real_trg.cpu())
                            mse_ = mse(x_fake_trg.cpu(), x_real_trg.cpu())
                            avg_psnr.update(psnr_)
                            avg_mse.update(mse_)

                            if ii == 0:
                                message = 'PSNR-01: {:4f} '.format(avg_psnr.avg)
                                message += 'MSE-01: {:4f} '.format(avg_mse.avg)
                            if ii == 1:
                                message = 'PSNR-02: {:4f} '.format(avg_psnr.avg)
                                message += 'MSE-02: {:4f} '.format(avg_mse.avg)
                            if ii == 2:
                                message = 'PSNR-10: {:4f} '.format(avg_psnr.avg)
                                message += 'MSE-10: {:4f} '.format(avg_mse.avg)
                            if ii == 3:
                                message = 'PSNR-12: {:4f} '.format(avg_psnr.avg)
                                message += 'MSE-12: {:4f} '.format(avg_mse.avg)
                            if ii == 4:
                                message = 'PSNR-20: {:4f} '.format(avg_psnr.avg)
                                message += 'MSE-20: {:4f} '.format(avg_mse.avg)
                            if ii == 5:
                                message = 'PSNR-21: {:4f} '.format(avg_psnr.avg)
                                message += 'MSE-21: {:4f} '.format(avg_mse.avg)
                            val_bar.set_description(desc=message)

                        psnr_mean_set[ii] = avg_psnr.avg
                        mse_mean_set[ii] = avg_mse.avg

                    # save all validate metrics 0-1 / 0-2 / 1-0 / 1-2 / 2-0 / 2-1
                    with open(os.path.join(self.sample_dir, 'vali_metrics.csv'), 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([i, 
                                         psnr_mean_set[0], mse_mean_set[0],
                                         psnr_mean_set[1], mse_mean_set[1],
                                         psnr_mean_set[2], mse_mean_set[2],
                                         psnr_mean_set[3], mse_mean_set[3],
                                         psnr_mean_set[4], mse_mean_set[4],
                                         psnr_mean_set[5], mse_mean_set[5]])

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'PET3':
            data_test_loader1 = self.pet3_test_loader1
            data_test_loader2 = self.pet3_test_loader2
            data_test_loader3 = self.pet3_test_loader3
            data_test_loader4 = self.pet3_test_loader4
            data_test_loader5 = self.pet3_test_loader5
            data_test_loader6 = self.pet3_test_loader6
        
        with torch.no_grad():
            for ii in range(0,6):
                if ii == 0:
                    val_bar = tqdm(data_test_loader1)
                if ii == 1:
                    val_bar = tqdm(data_test_loader2)
                if ii == 2:
                    val_bar = tqdm(data_test_loader3)
                if ii == 3:
                    val_bar = tqdm(data_test_loader4)
                if ii == 4:
                    val_bar = tqdm(data_test_loader5)
                if ii == 5:
                    val_bar = tqdm(data_test_loader6)

                avg_psnr = AverageMeter()
                avg_mse = AverageMeter()

                for nn, (x_real_org, x_real_trg, label_org, label_trg, x_MR) in enumerate(val_bar):
                    # Prepare input images and target domain labels.
                    x_real_org = x_real_org.to(self.device)
                    x_MR = x_MR.to(self.device)

                    c_org = self.label2onehot(label_org, self.c_dim)
                    c_trg = self.label2onehot(label_trg, self.c_dim)
                    c_org = c_org.to(self.device)
                    c_trg = c_trg.to(self.device) 

                    # Translate images.
                    inp = x_real_org
                    if self.use_MR:
                        inp = torch.cat([inp, x_MR], 1)
                    x_fake_trg = self.G(inp, c_trg)

                    # Calculate metrics
                    psnr_ = psnr(x_fake_trg.cpu(), x_real_trg.cpu())
                    mse_ = mse(x_fake_trg.cpu(), x_real_trg.cpu())
                    avg_psnr.update(psnr_)
                    avg_mse.update(mse_)

                    if ii == 0:
                        message = 'PSNR-01: {:4f} '.format(avg_psnr.avg)
                        message += 'MSE-01: {:4f} '.format(avg_mse.avg)
                    if ii == 1:
                        message = 'PSNR-02: {:4f} '.format(avg_psnr.avg)
                        message += 'MSE-02: {:4f} '.format(avg_mse.avg)
                    if ii == 2:
                        message = 'PSNR-10: {:4f} '.format(avg_psnr.avg)
                        message += 'MSE-10: {:4f} '.format(avg_mse.avg)
                    if ii == 3:
                        message = 'PSNR-12: {:4f} '.format(avg_psnr.avg)
                        message += 'MSE-12: {:4f} '.format(avg_mse.avg)
                    if ii == 4:
                        message = 'PSNR-20: {:4f} '.format(avg_psnr.avg)
                        message += 'MSE-20: {:4f} '.format(avg_mse.avg)
                    if ii == 5:
                        message = 'PSNR-21: {:4f} '.format(avg_psnr.avg)
                        message += 'MSE-21: {:4f} '.format(avg_mse.avg)
                    val_bar.set_description(desc=message)

                    # Save into nii for future analysis
                    if ii == 0:
                        folder_name = '01'
                    if ii == 1:
                        folder_name = '02'
                    if ii == 2:
                        folder_name = '10'
                    if ii == 3:
                        folder_name = '12'
                    if ii == 4:
                        folder_name = '20'
                    if ii == 5:
                        folder_name = '21'

                    result_path = os.path.join(self.result_dir, folder_name)
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)

                    x_real_org_nib = nib.Nifti1Image(x_real_org.cpu().numpy()[0,0,:,:,:], affine=np.eye(4))
                    x_fake_trg_nib = nib.Nifti1Image(x_fake_trg.cpu().numpy()[0,0,:,:,:], affine=np.eye(4))
                    x_real_trg_nib = nib.Nifti1Image(x_real_trg.cpu().numpy()[0,0,:,:,:], affine=np.eye(4))

                    nib.save(x_real_org_nib, os.path.join(result_path, str(nn) + '_real_org.nii.gz'))
                    nib.save(x_fake_trg_nib, os.path.join(result_path, str(nn) + '_fake_trg.nii.gz'))
                    nib.save(x_real_trg_nib, os.path.join(result_path, str(nn) + '_real_trg.nii.gz'))





