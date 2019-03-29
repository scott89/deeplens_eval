from network.network import Network
import tensorflow as tf

class KernelNet(Network):
    def __init__(self, input, is_training, trainable=True):
        self.is_training = is_training
        super(KernelNet, self).__init__(input, trainable)

    def setup(self):
        (self.feed('image', 'depth')
        .concat(3, name='im_depth')
        .conv(7, 7, 64, 1, 1, 1, biased=False, relu=False, name='conv1_shallow')
        .batch_normalization(relu=True, name='bn1_shallow', is_training=self.is_training))

        (self.feed('depth')
         .conv(7, 7, 64, 4, 4, 1, biased=False, relu=False, name='conv1')
         .batch_normalization(relu=True, name='bn1', is_training=self.is_training))
         #.max_pool(3, 3, 2, 2, name='pool1'))

        # res2a 80 128
        (self.conv(1, 1, 128, 1, 1, 1, biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(relu=True, name='bn2a_branch1', is_training=self.is_training))
        (self.feed('bn1')
         .conv(1, 1, 64, 1, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(relu=True, name='bn2a_branch2a', is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(relu=True, name='bn2a_branch2b', is_training=self.is_training)
         .conv(1, 1, 128, 1, 1, 1, biased=False, relu=False, name='res2a_branch2c')
         .batch_normalization(relu=False, name='bn2a_branch2c', is_training=self.is_training)
         .feed('bn2a_branch1', 'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu'))

        # res2b 40 256
        (self.conv(1, 1, 256, 2, 2, 1, biased=False, relu=False, name='res2b_branch1')
         .batch_normalization(relu=True, name='bn2b_branch1', is_training=self.is_training))
        (self.feed('res2a_relu')
         .conv(1, 1, 64, 1, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(relu=True, name='bn2b_branch2a', is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(relu=True, name='bn2b_branch2b', is_training=self.is_training)
         .conv(1, 1, 256, 2, 2, 1, biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(relu=False, name='bn2b_branch2c', is_training=self.is_training)
         .feed('bn2b_branch1', 'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu'))

        # res3a 40 256
        (self.conv(1, 1, 64, 1, 1, 1, biased=False, relu=False, name='res3a_branch2a')
         .batch_normalization(relu=True, name='bn3a_branch2a', is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(relu=True, name='bn3a_branch2b', is_training=self.is_training)
         .conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='res3a_branch2c')
         .batch_normalization(relu=False, name='bn3a_branch2c', is_training=self.is_training)
         .feed('res2b_relu', 'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu'))

        # output_down receptive_filed = 59
        #(self.conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='output_down_a')
        # .batch_normalization(relu=True, name='output_bn_down_a', is_training=self.is_training))

        # upsample
        self.resize(shape=tf.shape(self.layers['image'])[1:3], name='upsample')

        #concat
        (self.feed('upsample',
                   'bn1_shallow')
         .concat(3, name='concat'))


        # res4a
        (self.conv(1, 1, 128, 1, 1, 1, biased=False, relu=False, name='res4a_branch1')
            .batch_normalization(relu=True, name='bn4a_branch1', is_training=self.is_training))
        (self.feed('concat')
         .conv(1, 1, 64, 1, 1, 1, biased=False, relu=False, name='res4a_branch2a')
         .batch_normalization(relu=True, name='bn4a_branch2a', is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(relu=True, name='bn4a_branch2b', is_training=self.is_training)
         .conv(1, 1, 128, 1, 1, 1, biased=False, relu=False, name='res4a_branch2c')
         .batch_normalization(relu=False, name='bn4a_branch2c', is_training=self.is_training)
         .feed('bn4a_branch1', 'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu'))

        # res4b
        (self.conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='res4b_branch1')
         .batch_normalization(relu=True, name='bn4b_branch1', is_training=self.is_training))
        (self.feed('res4a_relu')
         .conv(1, 1, 64, 1, 1, 1, biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(relu=True, name='bn4b_branch2a', is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(relu=True, name='bn4b_branch2b', is_training=self.is_training)
         .conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='res4b_branch2c')
         .batch_normalization(relu=False, name='bn4b_branch2c', is_training=self.is_training)
         .feed('bn4b_branch1', 'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu'))
        # res4c
        #(self.conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='res4c_branch1')
        # .batch_normalization(relu=True, name='bn4c_branch1', is_training=self.is_training))
        (self.feed('res4b_relu')
         .conv(1, 1, 64, 1, 1, 1, biased=False, relu=False, name='res4c_branch2a')
         .batch_normalization(relu=True, name='bn4c_branch2a', is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res4c_branch2b')
         .batch_normalization(relu=True, name='bn4c_branch2b', is_training=self.is_training)
         .conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='res4c_branch2c')
         .batch_normalization(relu=False, name='bn4c_branch2c', is_training=self.is_training)
         .feed('res4b_relu', 'bn4c_branch2c')
         .add(name='res4c')
         .relu(name='res4c_relu'))
        (self.conv(1, 1, 31, 1, 1, 1, biased=True, relu=False, name='output')
         .relu(name='output_relu'))




