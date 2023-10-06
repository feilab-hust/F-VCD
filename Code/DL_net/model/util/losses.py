import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorflow.keras import backend as K


__all__ = ['mse_loss',
           'mae_loss',
           'HuberLoss',
           'huberMix_loss',
           'edge_loss',
           ]



def mse_loss(image, reference):
    with tf.variable_scope('l2_loss'):
        return tl.cost.mean_squared_error(image, reference,is_mean=True)
def mae_loss(image, reference):
    with tf.variable_scope('l1_loss'):
        return tl.cost.absolute_difference_error(image,reference,is_mean=True)

def HuberLoss(image,reference):
    loss = tf.reduce_mean(tf.losses.huber_loss(image, reference))
    return loss

def huberMix_loss(image,reference):

    loss = 1.0 * (
                0.5 * tl.cost.mean_squared_error(target=reference, output=image, is_mean=True)
                + 0.5 * tl.cost.absolute_difference_error(target=reference, output=image,is_mean=True))
    return loss


def edge_loss(image, reference):

    '''
    params:
        -image : tensor of shape [batch, depth, height, width, channels], the output of DVSR
        -reference : same shape as the image
    '''

    with tf.variable_scope('edges_loss'):
        edges_sr = tf.image.sobel_edges(image)
        edges_hr = tf.image.sobel_edges(reference)
        return mse_loss(edges_sr, edges_hr)


def SSIM_loss(image, reference,max_v=1.0,filter_size=5,filter_sigma=0.8):
    #return tf.reduce_mean(1-tf.image.ssim(image,reference,max_val=max_v))
    batch_size, H, W, z_depth = image.get_shape().as_list()
    loss_ssim=0

    for i in range(z_depth):
        # y1 =reference[..., i:i + 1]
        # y2 =image[..., i:i + 1]
        y1 = K.tile((reference[..., i:i + 1]), [1, 1, 1, 3])
        y2 = K.tile((image[..., i:i + 1]), [1, 1, 1, 3])
        temp = tf.reduce_mean(1-tf.image.ssim(y1,y2,max_val=max_v))
        loss_ssim = temp + loss_ssim
    loss_ssim = loss_ssim / z_depth
    return loss_ssim
    # return tf.reduce_mean(1-tf.image.ssim(image,reference,max_val=max_v))

