import argparse

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='DAMG: Degradation-Aware Multi-Prior Guidance')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--pretrained_weights', type=str, default='', help='optional checkpoint path used only for initialization before training')
    parser.add_argument('--resume_path', type=str, default='', help='optional training-state checkpoint used for resume; also supports legacy epoch_*.pth model checkpoints')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')
    parser.add_argument('--seed', type=int, default=3407, help='random seed for reproducible training')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_test_fivek'         , type=str, default='./datasets/FiveK/test/input')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # training objective profile
    parser.add_argument('--objective_profile', type=str, default='balanced',
    choices=['balanced', 'psnr', 'target29'],
    help='balanced keeps the current behavior; psnr and target29 bias training toward higher metric fidelity')
    parser.add_argument('--target_psnr', type=float, default=29.0, help='target PSNR used for goal-aware checkpoint selection')
    parser.add_argument('--target_ssim', type=float, default=0.90, help='target SSIM used for goal-aware checkpoint selection')
    parser.add_argument('--target_lpips', type=float, default=0.05, help='target LPIPS used for goal-aware checkpoint selection')
    parser.add_argument('--save_best_target', type=_str2bool, default=True, help='save the checkpoint that is closest to the target metric triplet')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=0.85)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)

    # evidence-arbitrated skeleton
    parser.add_argument('--use_prior_skeleton', type=_str2bool, default=True)
    parser.add_argument('--prior_mode', type=str, default='placeholder', choices=['placeholder', 'real', 'none'])
    parser.add_argument('--return_aux', type=_str2bool, default=True)
    parser.add_argument('--semantic_in_channels', type=int, default=3)
    parser.add_argument('--geometry_in_channels', type=int, default=4)
    parser.add_argument('--semantic_prior_dir', type=str, default='')
    parser.add_argument('--depth_prior_dir', type=str, default='')
    parser.add_argument('--normal_prior_dir', type=str, default='')
    parser.add_argument('--semantic_quality_dir', type=str, default='')
    parser.add_argument('--depth_quality_dir', type=str, default='')
    parser.add_argument('--normal_quality_dir', type=str, default='')
    parser.add_argument('--eval_semantic_prior_dir', type=str, default='')
    parser.add_argument('--eval_depth_prior_dir', type=str, default='')
    parser.add_argument('--eval_normal_prior_dir', type=str, default='')
    parser.add_argument('--eval_semantic_quality_dir', type=str, default='')
    parser.add_argument('--eval_depth_quality_dir', type=str, default='')
    parser.add_argument('--eval_normal_quality_dir', type=str, default='')
    parser.add_argument('--strict_prior_loading', type=_str2bool, default=True)
    parser.add_argument('--obs_loss_weight', type=float, default=0.18)
    parser.add_argument('--unobs_loss_weight', type=float, default=0.28)
    parser.add_argument('--prior_reg_weight', type=float, default=0.12)
    parser.add_argument('--route_entropy_weight', type=float, default=0.003)
    parser.add_argument('--completion_consistency_weight', type=float, default=0.08)
    parser.add_argument('--base_rgb_loss_weight', type=float, default=0.45)
    parser.add_argument('--branch_specialization_weight', type=float, default=0.15)
    parser.add_argument('--rgb_ssim_weight', type=float, default=0.0, help='weight for full-image RGB SSIM loss on final output')
    parser.add_argument('--base_rgb_ssim_weight', type=float, default=0.0, help='weight for RGB SSIM loss on pre-refined RGB output')
    parser.add_argument('--mask_focal_weight', type=float, default=1.0)
    parser.add_argument('--mask_dice_weight', type=float, default=0.5)
    parser.add_argument('--mask_l1_weight', type=float, default=0.25)
    parser.add_argument('--hvi_hv_weight', type=float, default=0.35)
    parser.add_argument('--hvi_i_weight', type=float, default=0.65)
    parser.add_argument('--head_refine_blocks', type=int, default=3)
    parser.add_argument('--rgb_refine_blocks', type=int, default=4)
    parser.add_argument('--lite_mode', type=str, default='none', choices=['none', 'safe'],
    help='none keeps the full DAMG estimator/fusion blocks; safe lightens them with separable convolutions')
    parser.add_argument('--lpips_train_weight', type=float, default=0.03)
    parser.add_argument('--lpips_train_resize', type=int, default=128)
    parser.add_argument('--perceptual_ramp_epochs', type=int, default=8)
    parser.add_argument('--lpips_ramp_epochs', type=int, default=16)
    parser.add_argument('--ssim_ramp_epochs', type=int, default=8)
    parser.add_argument('--detail_ramp_epochs', type=int, default=20)
    parser.add_argument('--completion_ramp_epochs', type=int, default=16)
    parser.add_argument('--color_contrast_weight', type=float, default=0.06, help='weight for local color/chroma contrast preservation')
    parser.add_argument('--reflectance_gradient_weight', type=float, default=0.04, help='weight for reflectance-gradient consistency')
    parser.add_argument('--texture_guard_weight', type=float, default=0.08, help='weight for texture/edge preservation in hard regions')
    parser.add_argument('--illumination_field_weight', type=float, default=0.06, help='weight for low-frequency illumination field consistency')
    parser.add_argument('--semantic_dropout_prob', type=float, default=0.08)
    parser.add_argument('--geometry_dropout_prob', type=float, default=0.06)
    parser.add_argument('--semantic_dropout_final_prob', type=float, default=0.02)
    parser.add_argument('--geometry_dropout_final_prob', type=float, default=0.015)
    parser.add_argument('--prior_noise_std', type=float, default=0.006)
    parser.add_argument('--prior_noise_final_std', type=float, default=0.0015)
    parser.add_argument('--prior_aug_decay_epochs', type=int, default=0, help='<=0 means auto schedule based on warmup epochs')
    parser.add_argument('--late_stage_start', type=int, default=240, help='epoch to begin fidelity-focused finetuning; <=0 disables')
    parser.add_argument('--late_stage_end', type=int, default=420, help='epoch to reach final late-stage loss scaling')
    parser.add_argument('--late_perceptual_scale', type=float, default=0.35, help='final multiplier for perceptual loss in late-stage finetuning')
    parser.add_argument('--late_lpips_scale', type=float, default=0.25, help='final multiplier for LPIPS loss in late-stage finetuning')
    parser.add_argument('--late_detail_scale', type=float, default=1.0, help='final multiplier for detail-oriented losses in late-stage finetuning')
    parser.add_argument('--late_rgb_ssim_scale', type=float, default=1.0, help='final multiplier for RGB SSIM loss in late-stage finetuning')
    parser.add_argument('--late_prior_reg_scale', type=float, default=0.55, help='final multiplier for prior regularization in late-stage finetuning')
    parser.add_argument('--late_route_reg_scale', type=float, default=0.80, help='final multiplier for route regularization in late-stage finetuning')
    parser.add_argument('--exposure_loss_weight', type=float, default=0.12, help='weight for global exposure/color alignment loss')
    parser.add_argument('--eval_tta', type=_str2bool, default=False)
    parser.add_argument('--eval_tta_mode', type=str, default='flip4', choices=['none', 'hflip', 'flip4'])
    parser.add_argument('--use_ema', type=_str2bool, default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=_str2bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=_str2bool, default=True, help='if gradient fluctuates too much, turn-on it')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='max gradient norm when grad_clip=True')
    
    
    # choose which dataset you want to train
    parser.add_argument('--dataset', type=str, default='lol_v1',
    choices=['lol_v1',
             'lolv2_real',
             'lolv2_syn',
             'lol_blur', 
             'SID',
             'SICE_mix',
             'SICE_grad',
             'fivek'],
    help='Select the dataset to train on (default: %(default)s)')

    return parser
