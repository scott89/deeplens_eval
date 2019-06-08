import tensorflow as tf
import numpy as np
#from network.resnet50 import ResNet50
#from network.resnet50_muti_task import ResNet50
from network.resnet50_lrelu import ResNet50

def normalize(x):
    x_min = tf.reduce_min(tf.reduce_min(x, axis=1, keep_dims=True), axis=2, keep_dims=True)
    x_max = tf.reduce_max(tf.reduce_max(x, axis=1, keep_dims=True), axis=2, keep_dims=True)
    return (x - x_min) / (x_max - x_min + np.finfo('float32').eps)

def build(im320_tensor, is_training):
    origin_size = tf.shape(im320_tensor)
    im_resized = tf.image.resize_bilinear(im320_tensor, [320, 320])
    with tf.variable_scope('Network'):
        with tf.variable_scope('Depth'):
            net = ResNet50({'data': im_resized-0.5}, is_training)
    pre = net.get_output()
    pre = tf.image.resize_bilinear(pre, [origin_size[1], origin_size[2]])
    pre = normalize(pre) 
    #pre = 1 - pre
    return pre, net
