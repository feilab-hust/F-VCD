import imageio
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time
from model import *
from utils import *
from config import config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


label = config.label
save_type=config.VALID.save_type
n_num = config.img_setting.Nnum
n_slices = config.img_setting.n_slices
sr_factor = config.img_setting.sr_factor
channels_interp = config.channels_interp
n_interp = config.n_interp
sub_pixel = config.sub_pixel
normalize_mode = config.preprocess.normalize_mode
if normalize_mode == 'normalize_percentile_z_score':
    normalize_fn = normalize_percentile_z_score
elif normalize_mode == 'percentile':
    normalize_fn = normalize_percentile
elif normalize_mode == 'constant':
    normalize_fn = normalize_constant
else:
    normalize_fn = normalize
print('normalize_mode: %s' % normalize_mode)


def __raise(info):
    raise Exception(info)

def ViewMap2LFP(input,n_num):
    batch,h,w,channel=input.shape
    base_h=h//n_num
    base_w=w//n_num
    assert channel==1,'wrong channel of net prediction'
    LFP_=np.zeros(input.shape,np.float32)
    for i in range(batch):
        for v in range(n_num):
            for u in range(n_num):
                LFP_[i,v::n_num, u::n_num,:]=input[:,v*base_h:(v+1)*base_h,u*base_w:(u+1)*base_w,:]
    return LFP_
def read_valid_images(path):
    """return images in shape [n_images, height=img_size/n_num, width=img_size/n_num, channels=n_num**2]
    """

    def __cast(im, dtype=np.float32):
        return im if im.dtype is np.float32 else im.astype(np.float32, casting='unsafe')

    img_list = sorted(tl.files.load_file_list(path=path, regx='.*.tif', printable=False))
    img_set = [__cast(get_and_rearrange3d(img_file, path, n_num=n_num, normalize_fn=normalize_fn)) for img_file in img_list]
    len(img_set) != 0 or __raise("none of the images have been loaded")
    print('read %d from %s' % (len(img_set), path))
    img_set = np.asarray(img_set)
    _, height, width, _ = img_set.shape

    return img_set, img_list, height, width


def infer(epoch, batch_size=1, use_cpu=False):
    """ Infer the 3-D images from the 2-D LF images using the trained VCD-Net

    Params:
        -epoch     : int, the epoch number of the checkpoint file to be loaded
        -batch_size: int, batch size of the VCD-Net
        -use_cpu   : bool, whether to use cpu for inference. If false, gpu will be used.
    """

    epoch = 'best' if epoch == 0 else epoch
    checkpoint_dir = config.Trans.ckpt_dir if args.trans else config.TRAIN.ckpt_dir
    valid_lr_img_path = config.VALID.lf2d_path
    save_dir = config.VALID.saving_path
    tl.files.exists_or_mkdir(save_dir)

    valid_lf_extras, names, height, width = read_valid_images(valid_lr_img_path)
    t_image = tf.placeholder('float32', [batch_size, height, width, 3])
    SR_size=config.img_setting.sr_factor*np.array([height,width])

    Recon_size=np.multiply(SR_size,config.img_setting.ReScale_factor)
    device_str = '/gpu:0'
    with tf.device(device_str):
        SR_net = LF_attention_denoise(LFP=t_image, output_size=SR_size, sr_factor=sr_factor, angRes=n_num,
                                  reuse=False, name=config.net_setting.SR_model,
                             channels_interp=64, normalize_mode=normalize_mode)

        Recon_net = MultiRes_UNeT_test(lf_extra=SR_net.outputs,
                                            n_slices=n_slices,
                                            output_size=Recon_size,
                                            is_train=True, reuse=False, name=config.net_setting.Recon_model,
                                            channels_interp=channels_interp, normalize_mode=normalize_mode
                                            )

    SR_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
                    ('.npz' in filename and epoch in filename and 'SR' in filename)]
    Recon_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
                       ('.npz' in filename and epoch in filename and 'recon' in filename)]


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)

        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, SR_ckpt_file[0]),
                                     network=SR_net)
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, Recon_ckpt_file[0]),
                                     network=Recon_net)


        # #im_buffer        = np.zeros([len(valid_lf_extras), height * n_num, width * n_num, config.PSF.n_slices])
        # im_buffer        = []
        # recon_start_time = time.time()

        recon_time = 0
        print('normlize_mode:%s net_tag:%s -- %s' % (normalize_mode, config.net_setting.SR_model,config.net_setting.Recon_model))

        for idx in range(0, len(valid_lf_extras), batch_size):
            start_time = time.time()
            sr_out = sess.run(SR_net.outputs, {t_image: valid_lf_extras[idx:idx + batch_size]})
            recon_out=sess.run(Recon_net.outputs, {t_image: valid_lf_extras[idx:idx + batch_size]})
            # if save_type=='LFP':
            #     sr_out=ViewMap2LFP(sr_out,n_num=n_num)
            batch_time = time.time() - start_time
            recon_time = recon_time + batch_time
            print("\rtime elapsed (sess.run): %4.4fs " % (time.time() - start_time), end='')


            temp_trans= np.transpose(np.transpose(np.squeeze(recon_out),[1,2,0]),[1,2,0])
            imageio.volwrite(save_dir + '%s' % (names[idx]),temp_trans)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('--trans', type=bool, default=False, help='')
    parser.add_argument("--cpu", help="use CPU instead of GPU for inference",
                        action="store_true")

    args = parser.parse_args()
    ckpt = args.ckpt
    batch_size = args.batch
    use_cpu = args.cpu

    infer(ckpt, batch_size=batch_size, use_cpu=use_cpu)



