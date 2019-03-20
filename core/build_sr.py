import tensorflow as tf
from network.srnet import SRNet
import numpy as np


def build(im320_tensor, depth320_tensor, dof320_tensor, im640_tensor, is_training):
    lr_input = tf.concat([im320_tensor, dof320_tensor, depth320_tensor], axis=3)
    hr_input = im640_tensor
    hr_shape = tf.shape(hr_input)
    dof_640_bicubic = tf.image.resize_images(tf.cast(dof320_tensor*255, tf.uint8), size=[hr_shape[1], hr_shape[2]], method=tf.image.ResizeMethod.BILINEAR)
    dof_640_bicubic = tf.to_float(dof_640_bicubic) / 255.0
    #dof_640_bicubic = tf.expand_dims(dof_640_bicubic, axis=0)
    with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('SR'):
            sr_net = SRNet({'lr_input': lr_input, 'hr_input': hr_input, 'dof_640': dof_640_bicubic}, is_training, bn_global=True)
        feature = sr_net.layers['fea_output']
        weight = sr_net.layers['pre1_5']

        feature = tf.stack([feature[:,:,:,::3], feature[:,:,:,1::3], feature[:,:,:,2::3]], axis=0)
        weight = weight / (tf.reduce_sum(weight, axis=3, keep_dims=True) + np.finfo("float").eps)
        pre_dof_640 = tf.reduce_sum(weight * feature, axis=4)
        pre_dof_640 = tf.transpose(pre_dof_640, [1,2,3,0])
        return pre_dof_640, sr_net

