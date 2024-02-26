from .util.utils import *
import tensorlayer as tl
import tensorflow as tf

def conv_block(layer, n_filter, kernel_size,
                is_train=True,
                activation=tf.nn.relu,is_in=False,
                border_mode="SAME",
                name='conv2d'):
    if is_in:
        s=conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, padding=border_mode,name=name+'_conv2d')
        s=instance_norm(s,name=name+'in',is_train=is_train)
        s.outputs=activation(s.outputs)
    else:
        s = conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, act=activation, padding=border_mode, name=name)
    return s
def MultiResBlock(layer,out_channel=None,is_train=True,alpha=1.0,name='MultiRes_block'):
    filter_num=out_channel*alpha
    n1_=int(filter_num*0.25)
    n2_=int(filter_num*0.25)
    n3_=int(filter_num*0.5)
    with tf.variable_scope(name):
        short_cut=layer
        short_cut=conv_block(short_cut,n_filter=n1_+n2_+n3_,kernel_size=1,is_train=is_train,is_in=True)
        conv3x3 = conv_block(layer, n_filter=n1_,kernel_size=3,is_train=is_train,is_in=True,name='conv_block1')
        conv5x5 = conv_block(conv3x3, n_filter=n2_,kernel_size=3,is_train=is_train,is_in=True,name='conv_block2')
        conv7x7 = conv_block(conv5x5, n_filter=n3_,kernel_size=3,is_train=is_train,is_in=True,name='conv_block3')
        out=concat([conv3x3,conv5x5,conv7x7],name='concat')
        out=instance_norm(out,is_train=is_train,name='in')
        out=merge([out,short_cut],name='merge_last')
        if out.outputs.get_shape().as_list()[-1] != out_channel:
            out=conv2d(out,n_filter=out_channel,filter_size=1,name='reshape_channel')
        out=LReluLayer(out,name='relu_last')
        out=instance_norm(out,name='batch_last')
    return out
def res_block(input, name='resblock'):
    c_num = input.outputs.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        n = conv2d(input, n_filter=c_num, filter_size=3, act=tf.nn.relu, name='conv1')
        n = conv2d(n, n_filter=c_num, filter_size=3, name='conv2')
        n = merge([input, n], name='res-add')
    return n

def ViewAttenionBlock(layer, out_channel, reduction=2, name='ViewAttenionBlock'):
    c_in = layer.outputs.get_shape().as_list()[-1]
    feature_num = c_in // reduction
    with tf.variable_scope(name):
        n1 = GlobalMeanPool2d(layer, keepdims=True, name='pooling1')
        n1 = conv2d(n1, filter_size=1, n_filter=feature_num, act=tf.nn.relu, name='FC1_1')
        n1 = conv2d(n1, filter_size=1, n_filter=c_in, act=tf.identity, name='FC1_2')

        n2 = GlobalMaxPool2d(layer, keepdims=True, name='pooling2')
        n2 = conv2d(n2, filter_size=1, n_filter=feature_num, act=tf.nn.relu, name='FC2_1')
        n2 = conv2d(n2, filter_size=1, n_filter=c_in, act=tf.identity, name='FC2_2')

        n = merge([n1, n2], name='merge')
        n.outputs = tf.nn.sigmoid(n.outputs)

        n = tl.layers.ElementwiseLayer([layer, n], combine_fn=tf.multiply, name='Attention')
        n = res_block(n, name='res_block1')
        n = res_block(n, name='res_block2')
    return n
def F_Denoise(LFP, output_size, angRes=11, sr_factor=7, upscale_mode='one_step', is_train=True, reuse=False,
                         name='unet', **kwargs):
    view_num = 3
    if 'channels_interp' in kwargs:
        channels_interp = kwargs['channels_interp']
    else:
        channels_interp = 64
    dialate_rate = [1, 2, 4]
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(LFP, 'MacronInput')
        n.outputs =  tf.transpose(n.outputs,perm=[3,1,2,0])
        with tf.variable_scope('Feature_extra'):
            n = conv2d(n, n_filter=channels_interp, filter_size=3, padding='SAME', name='conv0')
            aspp_pyramid = []
            for ii, d_r in enumerate(dialate_rate):
                feature = conv2d_dilate(n, n_filter=channels_interp, filter_size=3, dilation=d_r,name='dialate_pyramid_%d' % ii)
                aspp_pyramid.append(feature)
            n = concat(aspp_pyramid, concat_dim=-1, name='dialation_concat')
            n = conv2d(n, n_filter=channels_interp, filter_size=1, padding='SAME', name='Pyramid_conv')
            n = res_block(n, name='res_1')
        with tf.variable_scope('Attention'):
            n.outputs = tf.transpose(n.outputs, perm=[3, 1, 2, 0])  # convert to c,base_h,base_w,view
            fv0 = n
            fv1 = ViewAttenionBlock(fv0, out_channel=view_num, name='Atten_B1')
            fv2 = ViewAttenionBlock(merge([fv0, fv1], 'add_2'), out_channel=view_num, name='Atten_B2')
            FVA = merge([fv0, fv1, fv2], 'add_V')
            n.outputs = tf.transpose(n.outputs, perm=[3, 1, 2, 0])  # convert to view,base_h,base_w,c
            fs0 = n
            fs1 = ViewAttenionBlock(fs0, out_channel=channels_interp, name='spAtten_B1')
            fs2 = ViewAttenionBlock(merge([fs0, fs1], 'spadd_2'), out_channel=channels_interp, name='spAtten_B2')
            FSA = merge([fs0, fs1, fs2], 'spadd_V')

        with tf.variable_scope('fuse'):
            FVA.outputs = tf.transpose(FVA.outputs, perm=[3, 1, 2, 0])
            fuse = concat([FVA, FSA], name='SV_concat', concat_dim=-1)
            fuse = conv2d(fuse, n_filter=channels_interp // 2, filter_size=1, padding='SAME', name='conv1')
            fuse = res_block(fuse, 'res_1')
            fuse = conv2d(fuse, n_filter=channels_interp // 2, filter_size=1, padding='SAME', name='conv2')
            fuse = res_block(fuse, 'res_2')
        with tf.variable_scope('upscale'):
            out = conv2d(fuse,n_filter=1 ,filter_size=1, padding='SAME', name='conv2')
            out.outputs = tf.transpose(out.outputs, perm=[3, 1, 2, 0])
            # out = Stack2SAI(out, angRes=angRes, name='convert2SAI')
        return out

def F_Recon(lf_extra, n_slices, output_size, is_train=True, reuse=False, name='unet',**kwargs):

    if 'channels_interp' in kwargs:
      channels_interp = kwargs['channels_interp']
    else:
      channels_interp=128
    if 'normalize_mode' in kwargs:
      normalize_mode = kwargs['normalize_mode']
    else:
      normalize_mode='percentile'

    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(lf_extra, 'lf_extra')
        n = conv2d(n, n_filter=channels_interp, filter_size=7, name='conv1')
        dialate_rate = [1,2,4]
        ## Up-scale input
        with tf.variable_scope('feture_exta'):
            aspp_pyramid=[]
            for ii,d_r in enumerate(dialate_rate):
                feature= conv2d_dilate(n,n_filter=channels_interp,filter_size=3,dilation=d_r,name='dialate_pyramid_%d'%ii)
                aspp_pyramid.append(feature)
            n = concat(aspp_pyramid,concat_dim=-1,name='dialation_concat')
            n = conv2d(n,n_filter=channels_interp,filter_size=1,padding='SAME',name='Pyramid_conv')
            # interp
            channels_interp = channels_interp / 3
            n = SubpixelConv2d(n, scale=3, name='interp/subpixel1')
            n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv1')
            n = conv2d(n, n_filter=channels_interp, filter_size=3, name='conv_interp')
            n = instance_norm(n, name='in_interp')
            n = LReluLayer(n, name='reul_interp')
        pyramid_channels = [128, 256, 512, 512, 512]
        encoder_layers = []
        with tf.variable_scope('encoder'):
            n = conv2d(n, n_filter=64, filter_size=3, stride=1, name='conv0')
            n = instance_norm(n, is_train=is_train, name='in_0')
            n = LReluLayer(n, name='reul0' )
            for idx, nc in enumerate(pyramid_channels):
                encoder_layers.append(n) # append n0, n1, n2, n3, n4 (but without n5)to the layers list
                n = MultiResBlock(n,out_channel=nc,is_train=is_train,name='Multires_block_%d'%idx)
                n = tl.layers.MaxPool2d(n, filter_size=(3 ,3), strides=(2 ,2), name='maxplool%d' % (idx + 1))
        nl = len(encoder_layers)
        with tf.variable_scope('decoder'):
            _, h, w, _ = encoder_layers[-1].outputs.shape.as_list()
            n = tl.layers.UpSampling2dLayer(n ,size=(h, w) ,is_scale=False, name = 'upsamplimg')

            for idx in range(nl - 1, -1, -1): # idx = 4,3,2,1,0
                if idx > 0:
                    _, h, w, _ = encoder_layers[idx - 1].outputs.shape.as_list()
                    out_size = (h, w)
                    out_channels = pyramid_channels[idx -1]
                else:
                    out_channels = n_slices
                print('decoder %d : %s' % (idx, str(n.outputs.get_shape())))
                en_layer =encoder_layers[idx]
                n = ConcatLayer([en_layer, n], concat_dim=-1, name='concat%d' % (nl - idx))
                n = conv2d(n, out_channels, filter_size=3, stride=1 ,name='conv%d' % (nl - idx + 1))
                n = LReluLayer(n, name='relu%d' % (nl - idx + 1))
                n = instance_norm(n, is_train=is_train, name='in%d' % (nl - idx + 1))
                n = tl.layers.UpSampling2dLayer(n ,size=out_size ,is_scale=False, name = 'upsamplimg%d' % (nl - idx + 1))
            if n.outputs.shape[1] != output_size[0]:
                n = UpSampling2dLayer(n, size=output_size, is_scale=False, name = 'resize_final')
            n = conv2d(n, n_slices, filter_size=3, stride=1,name='conv_final' )
            if normalize_mode=='max':
                n.outputs = tf.tanh(n.outputs)
            return n

