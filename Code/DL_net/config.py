from easydict import EasyDict as edict
import os
config = edict()
config.img_setting = edict()
config.preprocess = edict()
config.net_setting = edict()
config.Pretrain=edict()
config.TRAIN = edict()
config.Loss = edict()
config.VALID = edict()

label='[Mito_test]_view3_L5c126'
#------------------------------image Setting----------------------------------
config.img_setting.img_size     = 160
config.img_setting.sr_factor=1
config.img_setting.ReScale_factor=[3,3]

config.img_setting.Nnum     = 3                                                             
config.img_setting.n_channels   = 1
config.channels_interp=126
config.n_interp = 1
config.sub_pixel = 3
config.img_setting.n_slices = 41

sourth_path=r'../../Data/Mito_View3_[LF160_SF160_WF480]/'
[Target3D,Synth_view,LFP] = [r'Stack/',r'HR/',r'LR/']
config.img_setting.Target3D       = os.path.join(sourth_path,Target3D)
config.img_setting.Synth_view     = os.path.join(sourth_path,Synth_view)
config.img_setting.LFP            = os.path.join(sourth_path,LFP)
#------------------------------Preprocess Setting----------------------------------
LFP_type='LFP'
SynView_type='LFP'
config.preprocess.normalize_mode = 'percentile'     #percentile; constant ; max
config.preprocess.LFP_type='1_%s'%(LFP_type)
config.preprocess.SynView_type='2_%s'%(SynView_type)
config.preprocess.discard_view=[]
#------------------------------net setting Setting----------------------------------
config.net_setting.SR_model='LF_attention'  #RDN,RCAN,TransAttenRcan,SA
config.net_setting.Recon_model='MultiRes_UNeT_test'  #Dynet,MultiRes
config.net_setting.is_bias=False
#------------------------------Label generate----------------------------------
config.label=r'%s_x%d]_[%s_%s]_%s'%(label,config.img_setting.sr_factor,config.net_setting.SR_model,config.net_setting.Recon_model,config.preprocess.normalize_mode)
#------------------------------Pretrain Setting----------------------------------
config.Pretrain.loading_pretrain_model=False
config.Pretrain.ckpt_dir=r'/'
#------------------------------Training Setting----------------------------------
config.TRAIN.test_saving_path     = "sample/test/{}/".format(config.label)
config.TRAIN.ckpt_saving_interval = 10
config.TRAIN.ckpt_dir             = "checkpoint/{}/".format(config.label)
config.TRAIN.log_dir              = "log/{}/".format(config.label)
config.TRAIN.valid_on_the_fly     = False

config.TRAIN.sample_ratio=1.0
config.TRAIN.shuffle_all_data=False
config.TRAIN.shuffle_for_epoch=True
config.TRAIN.device=0

config.TRAIN.batch_size  = 1
config.TRAIN.lr_init     = 1e-4
config.TRAIN.beta1       = 0.9
config.TRAIN.n_epoch     = 101
config.TRAIN.lr_decay    = 0.5
config.TRAIN.decay_every = 50

#---------------Loss Settings-----------------

config.Loss.Ratio = [0.5,0.5]
config.Loss.SR_loss={'mse_loss':1.0,
                     'mae_loss':0.1,
                     }
config.Loss.Recon_loss={'mse_loss':1.0,
                        'edge_loss':0.1,
                    }

#---------------validation settings-----------------
config.VALID.lf2d_path = r'../../Data/validation/'
config.VALID.save_type='LFP'
config.VALID.saving_path          = '{}SR_{}/'.format(config.VALID.lf2d_path,label)