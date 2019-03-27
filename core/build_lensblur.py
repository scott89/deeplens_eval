import tensorflow as tf
from network.kernel_net import KernelNet
from network.feature_net import FeaNet
import numpy as np

def _render_dof_batch(image_tensor, kernel_tensor, sigma=0.4):
    shifted_image = tf.extract_image_patches(image_tensor, [1, 17, 17, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
    shape = tf.shape(shifted_image)
    shifted_image = tf.reshape(shifted_image, [-1, shape[1], shape[2], 289, 3])
    shifted_image = tf.transpose(shifted_image, [4, 0, 1, 2, 3])
    image_max = tf.reduce_max(shifted_image, axis=0)
    weight = tf.exp(-(image_max-1)**2 / (2.0 * sigma**2))
    kernel_weight = kernel_tensor * weight
    #kernel_weight = kernel_weight / (tf.reduce_sum(kernel_weight, axis=3, keep_dims=True) + np.finfo("float").eps)
    dof_image = tf.reduce_sum(kernel_weight * shifted_image, axis=4)
    dof_image = tf.transpose(dof_image, [1,2,3,0])
    return dof_image



def build(im320_tensor, depth320_tensor, is_training):
    with tf.variable_scope('Network'):
        with tf.variable_scope('Lensblur'):
            lens_net = KernelNet({'image': im320_tensor-0.5, 'depth': depth320_tensor}, is_training)
            kernel = lens_net.get_output()
        with tf.variable_scope('Feature'):
            fea_net = FeaNet({'image': im320_tensor}, is_training)
            feature = fea_net.get_output()
    feature = tf.stack([feature[:,:,:,::3], feature[:,:,:,1::3], feature[:,:,:,2::3]], axis=0)
    kernel = kernel / (tf.reduce_sum(kernel, axis=3, keep_dims=True) + np.finfo("float").eps)
    dof_320 = tf.reduce_sum(kernel * feature, axis=4)
    dof_320 = tf.transpose(dof_320, [1,2,3,0])
    #dof320 = _render_dof_batch(im320_tensor, kernel)
    return dof_320, lens_net, fea_net

