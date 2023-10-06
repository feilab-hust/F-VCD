from .util.utils import *
import tensorlayer as tl
import tensorflow as tf


def conv_block(layer, n_filter, kernel_size,
               is_train=True,
               activation=tf.nn.relu, is_norm=False,
               border_mode="SAME",
               name='conv2d'):
    if is_norm:
        s = conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, padding=border_mode,
                   name=name + '_conv2d')
        s = batch_norm(s, name=name + 'in', is_train=is_train)
        s.outputs = activation(s.outputs)
    else:
        s = conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, act=activation, padding=border_mode,
                   name=name)
    return s


# def view_attention():


def MultiResBlock(layer, out_channel=None, is_train=True, alpha=1.0, name='MultiRes_block'):
    filter_num = out_channel * alpha
    n1_ = int(filter_num * 0.25)
    n2_ = int(filter_num * 0.25)
    n3_ = int(filter_num * 0.5)
    with tf.variable_scope(name):
        short_cut = layer
        short_cut = conv_block(short_cut, n_filter=n1_ + n2_ + n3_, kernel_size=1, is_train=is_train, is_norm=True)
        conv3x3 = conv_block(layer, n_filter=n1_, kernel_size=3, is_train=is_train, is_norm=True, name='conv_block1')
        conv5x5 = conv_block(conv3x3, n_filter=n2_, kernel_size=3, is_train=is_train, is_norm=True, name='conv_block2')
        conv7x7 = conv_block(conv5x5, n_filter=n3_, kernel_size=3, is_train=is_train, is_norm=True, name='conv_block3')
        out = concat([conv3x3, conv5x5, conv7x7], 'concat')
        out = batch_norm(out, is_train=is_train, name='in')
        out = merge([out, short_cut], name='merge_last')
        if out.outputs.get_shape().as_list()[-1] != out_channel:
            out = conv2d(out, n_filter=out_channel, filter_size=1, name='reshape_channel')
        out = LReluLayer(out, name='relu_last')
        out = batch_norm(out, name='batch_last')
    return out


def LF2SAI(input_tensor, angRes):
    # LFP realign
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(input_tensor[:, i::angRes, j::angRes, :])
        out.append(tf.concat(out_h, 2))
    out = tf.concat(out, 1)
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


def Macron2Stack(x, angRes, view_first=True):
    b, c, h, w = x.shape
    assert b == 1, 'angle attention needs to reshape tensor to c,view_num,base_h,base_w'
    out = []
    for i in range(angRes):
        for j in range(angRes):
            out.append(x[:, i::angRes, j::angRes, :])
    out = tf.concat(out, 0)  # view,base_h,base_w,c
    if not view_first:
        out = tf.transpose(out, [3, 1, 2, 0])
    return out


def LF_attention_denoise(LFP, output_size, angRes=11, sr_factor=7, upscale_mode='one_step', is_train=True, reuse=False,
                         name='unet', **kwargs):
    view_num = 3
    if 'channels_interp' in kwargs:
        channels_interp = kwargs['channels_interp']
    else:
        channels_interp = 64

    dialate_rate = [1, 2, 4]
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(LFP, 'MacronInput')
        # n.outputs = Macron2Stack(n.outputs, angRes=angRes, view_first=True)  # convert to view,base_h,base_w,c
        n.outputs =  tf.transpose(n.outputs,perm=[3,1,2,0])
        # n.outputs = tf.transpose(c_first,perm=[3,1,2,0])
        # longskip = n
        with tf.variable_scope('Feature_extra'):
            n = conv2d(n, n_filter=channels_interp, filter_size=3, padding='SAME', name='conv0')
            aspp_pyramid = []
            for ii, d_r in enumerate(dialate_rate):
                feature = conv2d_dilate(n, n_filter=channels_interp, filter_size=3, dilation=d_r,name='dialate_pyramid_%d' % ii)
                aspp_pyramid.append(feature)

            n = concat(aspp_pyramid, concat_dim=-1, name='dialation_concat')
            n = conv2d(n, n_filter=channels_interp, filter_size=1, padding='SAME', name='Pyramid_conv')
            n = res_block(n, name='res_1')
            # n = res_block(n, name='res_2')

        with tf.variable_scope('Attention'):
            n.outputs = tf.transpose(n.outputs, perm=[3, 1, 2, 0])  # convert to c,base_h,base_w,view
            fv0 = n
            fv1 = ViewAttenionBlock(fv0, out_channel=view_num, name='Atten_B1')
            fv2 = ViewAttenionBlock(merge([fv0, fv1], 'add_2'), out_channel=view_num, name='Atten_B2')
            # fv3 = ViewAttenionBlock(merge([fv0,fv1,fv2],'add_3'), out_channel=view_num, name='Atten_B3')
            # fv4 = ViewAttenionBlock(merge([fv0,fv1,fv2,fv3],'add_4'), out_channel=view_num, name='Atten_B4')
            FVA = merge([fv0, fv1, fv2], 'add_V')

            n.outputs = tf.transpose(n.outputs, perm=[3, 1, 2, 0])  # convert to view,base_h,base_w,c
            fs0 = n
            fs1 = ViewAttenionBlock(fs0, out_channel=channels_interp, name='spAtten_B1')
            fs2 = ViewAttenionBlock(merge([fs0, fs1], 'spadd_2'), out_channel=channels_interp, name='spAtten_B2')
            # fs3 = ViewAttenionBlock(merge([fs0, fs1, fs2], 'spadd_3'), out_channel=channels_interp, name='spAtten_B3')
            # fs4 = ViewAttenionBlock(merge([fs0, fs1, fs2, fs3], 'spadd_4'), out_channel=channels_interp, name='spAtten_B4')
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
