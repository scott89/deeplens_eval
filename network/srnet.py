from network.network import Network
import tensorflow as tf

class SRNet(Network):
    def __init__(self, input, is_training, trainable=True, bn_global=None, upsample_size=640):
        self.is_training = is_training
        self.bn_global = bn_global
        self.upsample_size = upsample_size
        super(SRNet, self).__init__(input, trainable)

    def setup(self):
        # lr_branch
        (self.feed('lr_input')
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='lr1_1')
        .batch_normalization(relu=True, name='lr1_1_bn', is_training=self.is_training)
        .conv(3, 3, 64, 2, 2, 1, biased=False, relu=False, name='lr2_1')
        .batch_normalization(relu=True, name='lr2_1_bn', is_training=self.is_training)
        .conv(3, 3, 128, 2, 2, 1, biased=False, relu=False, name='lr3_1')
        .batch_normalization(relu=True, name='lr3_1_bn', is_training=self.is_training)
        .conv(3, 3, 256, 2, 2, 1, biased=False, relu=False, name='lr4_1')
        .batch_normalization(relu=True, name='lr4_1_bn', is_training=self.is_training)
        .conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='lr4_2')
        .batch_normalization(relu=True, name='lr4_2_bn', is_training=self.is_training)
        .conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='lr5_1')
        .batch_normalization(relu=True, name='lr5_1_bn', is_training=self.is_training)
        .resize(shape=tf.shape(self.layers['lr3_1_bn'])[1:3], name='lr5_1_up')
        .feed('lr3_1_bn', 'lr5_1_up')
        .add(name='lr5_1_add')
        .conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='lr6_1')
        .batch_normalization(relu=True, name='lr6_1_bn', is_training=self.is_training)
        .resize(shape=tf.shape(self.layers['lr2_1'])[1:3], name='lr6_1_up')
        .feed('lr2_1_bn', 'lr6_1_up')
        .add(name='lr6_1_add')
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='lr7_1')
        .batch_normalization(relu=True, name='lr7_1_bn', is_training=self.is_training)
        .resize(shape=tf.shape(self.layers['lr1_1'])[1:3], name='lr7_1_up')
        .feed('lr1_1_bn', 'lr7_1_up')
        .add(name='lr7_1_add')
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='lr8_1')
        .batch_normalization(relu=True, name='lr8_1_bn', is_training=self.is_training)
        .resize(shape=tf.shape(self.layers['hr_input'])[1:3], name='lr8_1_up')
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='lr9_1')
        .batch_normalization(relu=True, name='lr9_1_bn', is_training=self.is_training)
        )

        # weight_prediction
        (self.feed('hr_input', 'lr9_1_bn')
        .concat(name='pre0')
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='pre1_1')
        .batch_normalization(relu=True, name='pre1_1_bn', is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='pre1_2')
        .batch_normalization(relu=True, name='pre1_2_bn', is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='pre1_3')
        .batch_normalization(relu=True, name='pre1_3_bn', is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, biased=False, relu=False, name='pre1_4')
        .batch_normalization(relu=True, name='pre1_4_bn', is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, biased=True, relu=True, name='pre1_5')
        )

        #image_feature
        (self.feed('dof_640', 'hr_input')
        .concat(name='fea0')
        .padding(padding=1, name='fea0-pad', mode='SYMMETRIC')
        .conv(3, 3, 30, 1, 1, 1, biased=True, relu=True, padding='VALID', name='fea1')
        .padding(padding=2, name='fea1-pad', mode='SYMMETRIC')
        .conv(5, 5, 30, 1, 1, 1, biased=True, relu=True, padding='VALID', name='fea2')
        .padding(padding=2, name='fea2-pad', mode='SYMMETRIC')
        .conv(5, 5, 15, 1, 1, 1, biased=True, relu=True, padding='VALID', name='fea3')
        .padding(padding=2, name='fea3-pad', mode='SYMMETRIC')
        .conv(5, 5, 15, 1, 1, 1, biased=True, relu=True, padding='VALID', name='fea4')
        .feed('fea0', 'fea1', 'fea2', 'fea3', 'fea4')
        .concat(name='fea_output')
        )
