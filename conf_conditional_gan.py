# MetaRelationNet 5 way 1 shot MiniImageNet
""" Network Options"""

net_conf = {}

net_conf['net_path'] = 'models.conditional_gan'

net_conf['lr_conf'] = {}
net_conf['lr_conf']['init_lr'] = 1e-5
net_conf['lr_conf']['decay_type'] = 'expo'
net_conf['lr_conf']['decay_base'] = 0.6
# if linearly decay:
# net_conf['lr_conf']['end_lr'] = 0
# net_conf['lr_conf']['start_decay_epoch'] = 0
# net_conf['lr_conf']['end_decay_epoch'] = 100

net_conf['optimizer_name'] = 'Adam'
net_conf['gan_type'] = 'gan'
net_conf['num_D_backward'] = 1 if net_conf['gan_type'] == 'wgan-gp' else 1
net_conf['generator_type'] = 'ResNet'
net_conf['use_perceptual'] = True
net_conf['use_feature_loss'] = False

net_conf['generator'] = {}
net_conf['generator']['num_in_chns'] = 3
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

## Options for solver
solver_conf = {}

solver_conf['solver_name'] = 'conditional_gan'
solver_conf['solver_path'] = 'models.conditional_gan'
solver_conf['gpu_ids'] = '1,0,2,3'
solver_conf['max_epoch'] = 100
solver_conf['checkpoints_dir'] = './checkpoints/res_net_kernel_5_gan_mse_perceptual_down_2x_3456_stride_2'
solver_conf['log_dir'] = './log_dir/res_net_kernel_5_gan_mse_perceptual_down_2x_3456_stride_2'
solver_conf['metric_names'] = ['psnr']
solver_conf['load_state'] = False
solver_conf['solver_state_path'] = './checkpoints/res_net_kernel_3_wgan_mse_SSIM_perceptual/network_saved.pkl'
solver_conf['load_epoch'] = False
solver_conf['phase'] = 'train'

# solver_conf['solver_state_path'] = '../model/MiniImageNet_MetaRelationNet_5way1shot_tune1/network_best.pkl'

## Options for data loader
loader_conf = {}
loader_conf['dataset_dir'] = '../../datasets/huawei_data2_registered'
loader_conf['train_imgs_desc_file'] = '../../datasets/huawei_data2_registered/HW_2_train_clean_crop.txt'
loader_conf['val_imgs_desc_file'] = '../../datasets/huawei_data2_registered/HW_2_val_clean_crop.txt'
loader_conf['crop_size'] = (3456, 2304)
loader_conf['batch_size'] = 4
loader_conf['num_workers'] = 6
loader_conf['pre_load'] = False

conf = {'net_conf':net_conf, 'solver_conf':solver_conf, 'loader_conf':loader_conf}
