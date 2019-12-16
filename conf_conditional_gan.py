# MetaRelationNet 5 way 1 shot MiniImageNet

## Options for solver
solver_conf = {}

solver_conf['solver_name'] = 'conditional_gan'
solver_conf['solver_path'] = 'models.conditional_gan'
solver_conf['gpu_ids'] = '1,0,2,3'
solver_conf['max_epoch'] = 100
solver_conf['checkpoints_dir'] = './checkpoints/iso50_3240_0.2_10_5_1_kernel_5555_stride_3333_dilation_2222'
solver_conf['log_dir'] = './log_dir/iso50_3240_0.2_10_5_1_kernel_5555_stride_3333_dilation_2222'
solver_conf['metric_names'] = ['psnr']
solver_conf['load_state'] = True
solver_conf['solver_state_path'] = './checkpoints/iso50_3240_0.2_10_5_1_kernel_5555_stride_3333_dilation_2222/network_14.pkl'
solver_conf['load_epoch'] = False
solver_conf['phase'] = 'val'
solver_conf['use_low_iso'] = True

# solver_conf['solver_state_path'] = '../model/MiniImageNet_MetaRelationNet_5way1shot_tune1/network_best.pkl'

""" Network Options"""
net_conf = {}

net_conf['net_path'] = 'models.conditional_gan'

net_conf['lr_conf'] = {}
net_conf['lr_conf']['init_lr'] = 1e-4
net_conf['lr_conf']['decay_type'] = 'expo'
net_conf['lr_conf']['decay_base'] = 0.9
# if linearly decay:
# net_conf['lr_conf']['end_lr'] = 0
# net_conf['lr_conf']['start_decay_epoch'] = 0
# net_conf['lr_conf']['end_decay_epoch'] = 100

net_conf['optimizer_name'] = 'Adam'
net_conf['gan_type'] = 'gan'
net_conf['num_D_backward'] = 0 if net_conf['gan_type'] == 'wgan-gp' else 0
net_conf['generator_type'] = 'ResNet'
if solver_conf['phase'] == 'val':
    net_conf['loss_weights'] = {'perceptual': 0, 'pix2pix': 0, 'ssim': 0, 'GAN': 0}
else:
    net_conf['loss_weights'] = {'perceptual': 0.2, 'pix2pix': 10, 'ssim': 5, 'GAN': 1}

net_conf['generator'] = {}
net_conf['generator']['num_in_chns'] = 6 if solver_conf['use_low_iso'] else 3
net_conf['generator']['num_out_chns'] = 3
net_conf['generator']['ngf'] = 64
net_conf['generator']['backbone_name'] = 'resnet_9blocks'
net_conf['generator']['norm_type'] = 'instance'
net_conf['generator']['use_dropout'] = True
net_conf['generator']['learn_residual'] = True

net_conf['generator']['kernel_size'] = 11

net_conf['discriminator'] = {}
net_conf['discriminator']['num_in_chns'] = 3
net_conf['discriminator']['num_out_chns'] = 3
net_conf['discriminator']['ndf'] = 64
net_conf['discriminator']['backbone_name'] = 'basic'
net_conf['discriminator']['num_layers'] = 3
net_conf['discriminator']['norm_type'] = 'instance'


## Options for data loader
loader_conf = {}
"""
loader_conf['dataset_dir'] = '../../datasets/huawei_data2_registered'
loader_conf['train_imgs_desc_file'] = '../../datasets/huawei_data2_registered/HW_2_train_clean_crop.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/huawei_data2_registered/HW_2_val_clean_crop.txt'
"""
"""
loader_conf['dataset_dir'] = '../../datasets/huawei_data2'
loader_conf['train_imgs_desc_file'] = '../../datasets/huawei_data2/train_clean.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/huawei_data2/val_clean.txt'
"""
"""
loader_conf['dataset_dir'] = '../../datasets/HW_3'
loader_conf['train_imgs_desc_file'] = '../../datasets/HW_3/train_clean.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/HW_3/val_clean.txt'
"""
"""
loader_conf['dataset_dir'] = '../../datasets/validation1/outdoor'
loader_conf['train_imgs_desc_file'] = '../../datasets/validation1/outdoor/test_outdoor.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/validation1/outdoor/test_outdoor.txt'
"""
"""
loader_conf['dataset_dir'] = '../../datasets/validation1/indoor'
loader_conf['train_imgs_desc_file'] = '../../datasets/validation1/indoor/test_indoor.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/validation1/indoor/test_indoor.txt'
"""
loader_conf['dataset_dir'] = '../../datasets/HW2_iso50'
loader_conf['train_imgs_desc_file'] = '../../datasets/HW2_iso50/train.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/HW2_iso50/val.txt'

# loader_conf['crop_size'] = (3240, 3240)
# loader_conf['crop_size'] = (5022, 5022)
# loader_conf['crop_size'] = (4608, 4608)
loader_conf['crop_size'] = (3402, 2268)
# loader_conf['crop_size'] = (3072, 2304)
loader_conf['batch_size'] = 4
loader_conf['num_workers'] = 6
loader_conf['pre_load'] = False
loader_conf['use_low_iso'] = solver_conf['use_low_iso']

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}
