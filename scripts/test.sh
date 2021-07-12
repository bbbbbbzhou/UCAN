CUDA_VISIBLE_DEVICES=0 python main.py \
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
