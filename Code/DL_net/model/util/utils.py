import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.layers import Layer
from config import config

w_init = tf.glorot_uniform_initializer  #Xavier uniform initializer.

if config.net_setting.is_bias:
    b_init = tf.constant_initializer(value=0.0)
else:
    b_init = None  #for previous unetzh
g_init = tf.random_normal_initializer(1., 0.02)


class LReluLayer(Layer):

    def __init__(self, layer=None, alpha=0.2, name='leaky_relu'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        with tf.variable_scope(name):
            self.outputs = tf.nn.leaky_relu(self.inputs, alpha=alpha)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
class ReluLayer(Layer):

    def __init__(self, layer=None, name='relu'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        with tf.variable_scope(name):
            self.outputs = tf.nn.relu(self.inputs)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class TanhLayer(Layer):

    def __init__(self, layer=None, name='Tanh_layer'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        with tf.variable_scope(name):
            self.outputs = tf.nn.tanh(self.inputs)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class GeluLayer(Layer):
    def __init__(self, layer=None, name='GeluLayer'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        with tf.variable_scope(name):
            x = self.inputs
            self.outputs =(0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0))))*x
            # self.outputs = 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


def conv2d(layer, n_filter, filter_size=3, stride=1, act=tf.identity, W_init=w_init,padding='SAME', b_init=b_init, name = 'conv2d'):
    return tl.layers.Conv2d(layer, n_filter=int(n_filter), filter_size=(filter_size, filter_size), strides=(stride, stride), act=act, padding=padding, W_init=W_init, b_init=b_init, name=name)


def conv2d_dilate(layer, n_filter, filter_size=3, stride=(1, 1), act=tf.identity, dilation=1, W_init=w_init,padding='SAME',
                  b_init=b_init, name='dilated_conv2d'):
    return tl.layers.AtrousConv2dLayer(layer, n_filter=int(n_filter), filter_size=(filter_size,filter_size), act=act,
                            padding=padding, rate=dilation, W_init=W_init, b_init=b_init, name=name)


def conv3d(layer,
    act=tf.identity, 
    filter_shape=(2,2,2,3,32),  #Shape of the filters: (filter_depth, filter_height, filter_width, in_channels, out_channels)
    strides=(1, 1, 1, 1, 1), W_init=w_init, b_init=b_init, name='conv3d'): 
    
    return tl.layers.Conv3dLayer(layer, act=act, shape=filter_shape, strides=strides, padding='SAME', W_init=W_init, b_init=b_init, W_init_args=None, b_init_args=None, name=name)

def deconv2d(layer, out_channels, filter_size=3, stride=2, out_size=None, act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='deconv2d'):
    """
    up-sampling the layer in height and width by factor 2
    Parames:
        shape - shape of filter : [height, width, out_channels, in_channels]
        out_size : height and width of the outputs 
    """
    batch, h, w, in_channels = layer.outputs.get_shape().as_list()   
    filter_shape = (filter_size, filter_size, int(out_channels), int(in_channels))
    if out_size is None:
        output_shape = (batch, int(h * stride), int(w * stride), int(out_channels))
    else :
        output_shape = (batch, out_size[0], out_size[1], int(out_channels))
    strides = (1, stride, stride, 1)
    return tl.layers.DeConv2dLayer(layer, act=act, shape=filter_shape, output_shape=output_shape, strides=strides, padding=padding, W_init=W_init, b_init=b_init, W_init_args=None, b_init_args=None, name=name)
def concat(layer, concat_dim=-1, name='concat'):
    return ConcatLayer(layer, concat_dim=concat_dim, name=name)

def atrous2d(layer, out_channels, filter_size, rate, act=tf.identity, padding='VALID', name='atrous2d'):
    return tl.layers.AtrousConv2dLayer(
                 prev_layer=layer,
                 n_filter=out_channels,
                 filter_size=(filter_size, filter_size),
                 rate=rate,
                 act=act,
                 padding=padding,
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_args=None,
                 b_init_args=None,
                 name=name)

def merge(layers, name='merge'):
    '''
    merge two Layers by element-wise addition
    Params : 
        -layers : list of Layer instances to be merged : [layer1, layer2, ...]
    '''
    return tl.layers.ElementwiseLayer(layers, combine_fn=tf.add, name=name)

def instance_norm(layer, act=tf.identity, is_train=True, gamma_init=g_init, name='IN'):
    return tl.layers.InstanceNormLayer(layer, act=act, name=name)
def batch_norm(layer, act=tf.identity, is_train=True, gamma_init=g_init, name='bn'): 
    return tl.layers.BatchNormLayer(layer, act=act, is_train=is_train, gamma_init=gamma_init, name=name)

def max_pool2d(layer, filter_size=2, stride=2, name='pooling3d'):
    return MaxPool2d(layer, filter_size=(filter_size, filter_size), strides=(stride, stride), name=name)

class PadDepth(Layer):
    
    def __init__(self, layer=None, name='padding',desired_channels=0):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.desired_channels=desired_channels
        
        with tf.variable_scope(name):
            self.outputs = self.pad_depth(self.inputs,self.desired_channels)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        
    def pad_depth(self, x , desired_channels):
        y = tf.zeros_like(x)
        print(y.shape)
        new_channels = desired_channels - x.shape.as_list()[-1]
        y = y[...,:new_channels]
        
        #y=tf.to_int32(y, name='ToInt32')
        #x=tf.to_int32(x, name='ToInt32')
        print(x.shape,y.shape)
        return tf.concat([x,y],axis=-1)                              


 
def UpConv(layer, out_channels, filter_size=4, factor=2, name='upconv'):
    with tf.variable_scope(name):
        n = tl.layers.UpSampling2dLayer(layer, size=(factor, factor), is_scale=True, method=1, name = 'upsampling')
        '''
        - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
        - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
        - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
        - Index 3 ResizeMethod.AREA, Area interpolation.
        '''
        n = conv2d(n, n_filter=out_channels, filter_size=filter_size, name='conv1')
        return n
def upconv(layer, out_channels, out_size, filter_size=3, name='upconv'):
    with tf.variable_scope(name):
        n = tl.layers.UpSampling2dLayer(layer, size=out_size, is_scale=False, method=1, name = 'upsampling')
        '''
        - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
        - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
        - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
        - Index 3 ResizeMethod.AREA, Area interpolation.
        '''
        n = conv2d(n, n_filter=out_channels, filter_size=filter_size, name='conv')
        return n
class Macron2Stack(Layer):
    def __init__(self, layer=None, name='Macron2StackLayer', n_num=11):

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # def _Macron2Stack(LF, n_num=11):
        #     batch, h, w, channel = LF.get_shape().as_list()
        #     assert channel == 1, 'wrong LF tensor input'
        #     LF_extra = []
        #     for b in range(batch):
        #         temp_extra = []
        #         lf_2d = LF[b, ...]
        #         for i in range(n_num):
        #             for j in range(n_num):
        #                 view_map = lf_2d[i: h: n_num, j: w: n_num]
        #                 temp_extra.append(view_map[tf.newaxis, ...])
        #         LF_extra.append(tf.concat(temp_extra, axis=-1))
        #     LF_extra = tf.concat(LF_extra, axis=0)
        #     return LF_extra

        self.outputs = tf.space_to_depth(self.inputs, n_num)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])

class SAI2ViewStack(Layer):
    def __init__(self, layer=None, name='Macron2StackLayer', n_num=11):

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        def _SAI2ViewStack(LF, angRes=11):
            batch, h, w, channel = LF.get_shape().as_list()
            base_h,base_w=h//angRes,w//angRes
            assert channel == 1, 'wrong LF tensor input'
            out = []
            for i in range(angRes):
                out_h = []
                for j in range(angRes):
                    out_h.append(LF[:, i*base_h:(i+1)*base_h, j*base_w:(j+1)*base_w, :])
                out.append(tf.concat(out_h, -1))
            out = tf.concat(out, -1)
            return out



        # self.outputs = tf.space_to_depth(self.inputs, n_num)
        self.outputs = _SAI2ViewStack(self.inputs, n_num)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])