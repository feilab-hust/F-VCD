import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
from model import *
from dataset import Dataset
from utils import write3d
from config import config

###=================img pre ===========================###
img_size = config.img_setting.img_size
n_num = config.img_setting.Nnum
sr_factor=config.img_setting.sr_factor
n_slices=config.img_setting.n_slices
ReScale_factor=config.img_setting.ReScale_factor
channels_interp = config.channels_interp
n_interp = config.n_interp
sub_pixel = config.sub_pixel

sample_ratio= config.TRAIN.sample_ratio

base_size = img_size // n_num  # lateral size of lf_extra
normalize_mode = config.preprocess.normalize_mode
test_num = 4

###=================training para ===========================###
loading_pretrain_model=config.Pretrain.loading_pretrain_model
pretrain_ckpt_dir=config.Pretrain.ckpt_dir

batch_size = config.TRAIN.batch_size
shuffle_for_epoch=config.TRAIN.shuffle_for_epoch
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

###=================dir ===========================###
label = config.label
test_saving_dir = config.TRAIN.test_saving_path
checkpoint_dir = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval
log_dir = config.TRAIN.log_dir
plot_test_loss_dir = os.path.join(test_saving_dir, 'test_loss_plt')
test_hr_dir = os.path.join(test_saving_dir, 'HR_View')
test_lf_dir = os.path.join(test_saving_dir, 'LFP')
test_stack_dir = os.path.join(test_saving_dir, 'Target3D')

###=================losses define ===========================###
SR_loss=config.Loss.SR_loss
Recon_loss=config.Loss.Recon_loss
loss_ratio=config.Loss.Ratio

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.losses = {}

    def build_graph(self):
        ###========================== DEFINE MODEL ============================###
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(lr_init, trainable=False)

        SR_tag=config.net_setting.SR_model
        Recon_tag=config.net_setting.Recon_model


        view_num= n_num
        input_size = np.array([img_size, img_size])
        SR_size=input_size*sr_factor
        Recon_size=np.multiply(SR_size,ReScale_factor)

        self.plchdr_lf = tf.placeholder('float32', [batch_size, *input_size, view_num],name='t_LFP')
        self.plchdr_SynView = tf.placeholder('float32', [batch_size, *SR_size, view_num],name='t_SynView')
        self.plchdr_target3d = tf.placeholder('float32', [batch_size, *Recon_size, n_slices],name='t_target3d')


        with tf.device('/gpu:{}'.format(config.TRAIN.device)):
            self.SR_net =  LF_attention_denoise(LFP=self.plchdr_lf,
                                                output_size=SR_size,
                                                sr_factor=sr_factor,
                                                angRes=n_num,reuse=False, name=SR_tag, channels_interp=64,normalize_mode=normalize_mode)

            self.Recon_net = MultiRes_UNeT_test(lf_extra=self.SR_net.outputs,
                                           n_slices=n_slices,
                                           output_size=Recon_size,
                                           is_train=True, reuse=False, name=config.net_setting.Recon_model,
                                           channels_interp=channels_interp, normalize_mode=normalize_mode
                                           )

        self.SR_net.print_params(False)
        self.Recon_net.print_params(False)

        SR_vars = tl.layers.get_variables_with_name(SR_tag, train_only=True, printable=False)
        Recon_vars= tl.layers.get_variables_with_name(Recon_tag, train_only=True, printable=False)
        # ====================
        # loss function
        # =====================
        self.loss = 0  # initial
        # self._get_losses()    # get losses
        self.SR_loss=0
        self.Recon_loss=0

        # define SR loss
        for key in SR_loss:
            temp_func=eval(key)
            temp_loss=temp_func(image=self.SR_net.outputs, reference=self.plchdr_SynView)
            self.SR_loss=self.SR_loss+SR_loss[key]*temp_loss
            self.losses.update({'SR_'+key: SR_loss[key]*temp_loss})
            tf.summary.scalar(key,temp_loss)
        # define Recon loss
        for key in Recon_loss:
            temp_func=eval(key)
            temp_loss=temp_func(image=self.Recon_net.outputs, reference=self.plchdr_target3d)
            self.Recon_loss = self.Recon_loss + Recon_loss[key] * temp_loss
            self.losses.update({'Recon_'+key: Recon_loss[key] * temp_loss})
            tf.summary.scalar(key,temp_loss)

        self.loss=loss_ratio[0]*self.SR_loss+loss_ratio[1]*self.Recon_loss
        tf.summary.scalar('learning_rate', self.learning_rate)
        # define test_loss when test
        self.loss_test = self.loss
        # ----------------create sess-------------
        configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configProto)

        # self.pre_train_opt =
        self.SR_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.SR_loss, var_list=SR_vars)
        self.Recon_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss, var_list=SR_vars+Recon_vars)
        self.merge_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

    def _train(self, begin_epoch):
        """Train the VCD-Net
        Params
            -begin_epoch: int, if not 0, a checkpoint file will be loaded and the training will continue from there
        """
        ## create folders to save result images and trained model


        save_dir = test_saving_dir
        tl.files.exists_or_mkdir(save_dir)
        # save_configs(save_folder=os.path.join(os.getcwd(),save_dir))
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(log_dir)
        tl.files.exists_or_mkdir(plot_test_loss_dir)
        tl.files.exists_or_mkdir(test_lf_dir)
        tl.files.exists_or_mkdir(test_hr_dir)
        tl.files.exists_or_mkdir(test_stack_dir)
        #initialize vars
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.learning_rate, lr_init))

        if loading_pretrain_model and begin_epoch==0:
            SR_ckpt_file = [filename for filename in os.listdir(pretrain_ckpt_dir) if
                         ('.npz' in filename and 'best' in filename and 'denoise' in filename)]
            Recon_ckpt_file= [filename for filename in os.listdir(pretrain_ckpt_dir) if
                         ('.npz' in filename and 'best' in filename and 'recon' in filename)]
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(pretrain_ckpt_dir, SR_ckpt_file[0]),
                                         network=self.SR_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(pretrain_ckpt_dir , Recon_ckpt_file[0]),
                                         network=self.Recon_net)
        if (begin_epoch != 0):
            SR_ckpt=self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir,epoch=begin_epoch,label='denoise')
            Recon_ckpt=self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir,epoch=begin_epoch,label='recon')
            assert SR_ckpt!=None and Recon_ckpt!=None,'No ckpt has been found'
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir, SR_ckpt),network=self.SR_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir, Recon_ckpt),network=self.Recon_net)

        ###====================== LOAD DATA ===========================###
        dataset_size = self.dataset.prepare(batch_size, n_epoch)
        final_cursor = (dataset_size // batch_size - 1) * batch_size
        self._get_test_data()

        fetches = self.losses
        fetches['opti_sr'] = self.SR_optim
        fetches['opti_recon'] = self.Recon_optim
        fetches['batch_summary'] = self.merge_op

        while self.dataset.hasNext():

            Stack_batch,HR_batch, LF_batch, cursor, epoch = self.dataset.iter()  # get data
            feed_train = {
                          self.plchdr_target3d: Stack_batch,
                          self.plchdr_SynView:HR_batch,
                          self.plchdr_lf: LF_batch,
                          }

            # if list(set(self.test_hr_list) & set(hr3d_list_batch)):
            #     raise Exception('training data and validation data overlap')

            epoch += begin_epoch
            step_time = time.time()

            # learning rate update
            if epoch != 0 and (epoch % decay_every == 0) and cursor == 0:
                new_lr_decay = lr_decay ** (epoch // decay_every)
                self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))

            # infer
            evaluated = self.sess.run(fetches, feed_train)
            # log
            loss_str = [name + ':' + str(value) for name, value in evaluated.items() if 'loss' in name]
            print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs ---%s" % (
                epoch, n_epoch+begin_epoch, cursor, dataset_size, time.time() - step_time, loss_str), end='')
            self.summary_writer.add_summary(evaluated['batch_summary'],
                                            epoch * (dataset_size // batch_size - 1) + cursor / batch_size)
            ##record and save checkpoints
            if cursor == final_cursor:
                self._record_avg_test_loss(epoch, self.sess)
                if epoch != 0 and (epoch % ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)

    def _get_test_data(self):
        self.test_target3d, self.test_Synview, self.test_LFP= self.dataset.for_test()
        for i in range(test_num):
            write3d(self.test_target3d[i:i + 1],test_stack_dir + '/Target3d_%d.tif' % i)
            write3d(self.test_Synview[i:i + 1],test_hr_dir + '/SynView_%d.tif' % i)
            write3d(self.test_LFP[i:i + 1],test_lf_dir + '/LFP_%d.tif' % i)

    def _save_intermediate_ckpt(self, tag, sess):
        tag = ('epoch%d' % tag) if is_number(tag) else tag
        sr_file_name = checkpoint_dir + '/SR_net_{}.npz'.format(tag)
        recon_file_name=checkpoint_dir + '/recon_net_{}.npz'.format(tag)
        tl.files.save_npz(self.SR_net.all_params, name=sr_file_name, sess=sess)
        tl.files.save_npz(self.Recon_net.all_params, name=recon_file_name, sess=sess)
        if 'epoch' in tag:
            if batch_size >= test_num:
                test_lr_batch = self.test_LFP[0:batch_size]
                SR_view = self.sess.run(self.SR_net.outputs, {self.plchdr_lf: test_lr_batch})
                Recon_stack= self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                for i in range(test_num):
                    write3d(SR_view[i:i + 1], test_saving_dir + ('SR_{}_%d.tif' % (i)).format(tag))
                    write3d(Recon_stack[i:i + 1], test_saving_dir + ('Recon_{}_%d.tif' % (i)).format(tag))
            else:
                for idx in range(0, test_num, batch_size):
                    if idx + batch_size <= test_num:
                        test_lr_batch = self.test_LFP[idx:idx + batch_size]
                        SR_view = self.sess.run(self.SR_net.outputs, {self.plchdr_lf: test_lr_batch})
                        Recon_stack = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                        for i in range(len(SR_view)):
                            write3d(SR_view[i:i + 1],test_saving_dir + ('SR_{}_%d.tif' % (i + idx * batch_size)).format(tag))
                            write3d(Recon_stack[i:i + 1],test_saving_dir + ('Recon_{}_%d.tif' % (i + idx * batch_size)).format(tag))
    def _record_avg_test_loss(self, epoch, sess):
        if 'min_test_loss' not in dir(self):
            self.min_test_loss = 1e10
            self.best_epoch = 0
            self.test_loss_plt = []

        test_loss = 0
        test_data_num = len(self.test_LFP)
        print("")
        for idx in range(0, test_data_num, batch_size):
            if idx + batch_size <= test_data_num:
                test_lf_batch = self.test_LFP[idx: idx + batch_size]
                test_synview__batch = self.test_Synview[idx: idx + batch_size]
                test_stack__batch = self.test_target3d[idx: idx + batch_size]
                feed_test = {self.plchdr_lf: test_lf_batch,self.plchdr_SynView:test_synview__batch, self.plchdr_target3d: test_stack__batch}
                test_loss_batch, losses_batch = sess.run([self.loss_test, self.losses], feed_test)
                loss_str = [name + ':' + str(value) for name, value in losses_batch.items() if 'loss' in name]
                test_loss += test_loss_batch
                print('\rvalidation  [% 2d/% 2d] loss = %.6f --%s ' % (idx, test_data_num, test_loss_batch, loss_str),
                      end='')
        test_loss /= (len(self.test_LFP) // batch_size)
        print('\navg = %.6f best = %.6f (@epoch%d)' % (test_loss, self.min_test_loss, self.best_epoch))
        self.test_loss_plt.append([epoch, test_loss])
        temp_file_name = plot_test_loss_dir + '/plot_test_loss.npy'
        np.save(temp_file_name, self.test_loss_plt)

        if (test_loss < self.min_test_loss):
            self.min_test_loss = test_loss
            self.best_epoch = epoch
            self._save_intermediate_ckpt(tag='best', sess=sess)
            # self._save_pb(sess)

    def _plot_test_loss(self):
        loss = np.asarray(self.test_loss_plt)
        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        plt.savefig(plot_test_loss_dir + '/test_loss.png', bbox_inches='tight')
        plt.show()


    def _traversal_through_ckpts(self, checkpoint_dir, epoch, label=None):
        ckpt_found = False
        filelist = os.listdir(checkpoint_dir)
        for file in filelist:
            if '.npz' in file and str(epoch) in file:
                if label is not None:
                    if label in file:
                        return file
                else:
                    return file
        return None

    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--ckpt', type=int, default=0, help='')
    parser.add_argument('-idx', '--model_idx', type=int, default='0', help='')
    parser.add_argument('--trans', type=bool, default=False, help='')
    args = parser.parse_args()
    training_dataset = Dataset( config.img_setting.Target3D,
                                config.img_setting.Synth_view,
                                config.img_setting.LFP,
                                n_num,
                                base_size,
                                shuffle_for_epoch=shuffle_for_epoch,
                                normalize_mode=normalize_mode,
                                sample_ratio=sample_ratio,
                                shuffle_all_data=config.TRAIN.shuffle_all_data)
    trainer = Trainer(training_dataset)
    trainer.build_graph()
    trainer.train(begin_epoch=args.ckpt)




