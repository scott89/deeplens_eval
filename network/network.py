import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'
DEFAULT_INIT = tf.contrib.layers.xavier_initializer()


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        self.regularizer = [0.0]
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        if param_name == 'mean':
                            param_name = 'moving_mean'
                        elif param_name == 'variance':
                            param_name = 'moving_variance'
                        elif param_name == 'scale':
                            param_name = 'gamma'
                        elif param_name == 'offset':
                            param_name = 'beta'
                        var = tf.get_variable(param_name)
                        if op_name + '/' + param_name == 'Output_new_full1/weights':
                            data = data[:,:,:,[1]]
                        elif op_name + '/' + param_name == 'Output_new_full1/biases':
                            data = data[[1]]
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, weight_decay=0, trainable=None, initializer=None):
        '''Creates a new TensorFlow variable.'''
        if trainable == None:
            trainable = self.trainable
        var = tf.get_variable(name, shape, trainable=trainable, initializer=initializer)
        if weight_decay != 0:
            self.regularizer.append(weight_decay * tf.nn.l2_loss(var))
            #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay * tf.nn.l2_loss(var))

        return var

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        try:
            assert padding in ('SAME', 'VALID')
        except:
            a=1

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             dilation_rate,
             name,
             weight_decay=5e-4,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape().as_list()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        if dilation_rate == 1:
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        else:
            assert s_h == 1 and s_w == 1
            convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation_rate, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o], weight_decay=weight_decay, initializer=DEFAULT_INIT)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input, group, axis=3)
                kernel_groups = tf.split(kernel, group, axis=3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(output_groups, 3)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output
    @layer
    def separable_conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             dilation_rate,
             name,
             c_m = 1,
             weight_decay=5e-4,
             relu=True,
             padding=DEFAULT_PADDING,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Convolution for a given input and kernel
        if dilation_rate > 1:
            assert s_h == 1 and s_w == 1
        separable_convolve = lambda i, k_depth, k_point: tf.nn.separable_conv2d(i, k_depth, k_point, strides=[1, s_h, s_w, 1],
                rate=[dilation_rate, dilation_rate], padding=padding, name=name)
        with tf.variable_scope(name) as scope:
            #kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            k_depth = self.make_var('kernel_depth', shape=[k_h, k_w, c_i, c_m], weight_decay=weight_decay, initializer=DEFAULT_INIT)
            k_point = self.make_var('kernel_point', shape=[1, 1, c_i*c_m, c_o], weight_decay=weight_decay, initializer=DEFAULT_INIT)
            output = separable_convolve(input, k_depth, k_point)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output


    @layer
    def deconv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             trainable=None,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        n_i, h_i, w_i, c_i = input.get_shape().as_list()
        assert c_i % group == 0
        assert c_o % group == 0
        if padding == 'SAME':
            p_h = np.int(np.floor((k_h - s_h) / 2.0))
            p_w = np.int(np.floor((k_w - s_w) / 2.0))
        else:
            p_h = p_w = np.int(0)

        h_o = (h_i - 1) * s_h + k_h - 2 * p_h
        w_o = (w_i - 1) * s_w + k_w - 2 * p_w
        # Convolution for a given input and kernel

        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape=[n_i, h_o, w_o, c_o/group], strides=[1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o], trainable=trainable)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = deconvolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input, group, axis=3)
                kernel_groups = tf.split(kernel, group, axis=3)
                output_groups = [deconvolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(output_groups, 3)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], trainable=trainable)
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis=3, name=None):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape().as_list()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, is_training=None, scale_offset=True, relu=False, lrelu=False, alpha=0.2,
            momentum=0.99, renorm=False, epsilon=1e-5):
        if is_training is None:
            is_training = self.is_training
        if hasattr(self, 'bn_global') and self.bn_global is not None:
            is_training = not self.bn_global
        output = tf.layers.batch_normalization(input, training=is_training, name=name, center=scale_offset, scale=scale_offset, momentum=momentum, epsilon=epsilon, renorm=renorm)
        # NOTE: Currently, only inference is supported
        if relu:
            with tf.variable_scope(name) as scope:
                output = tf.nn.relu(output)
        elif lrelu:
            with tf.variable_scope(name) as scope:
                output = tf.nn.leaky_relu(output, alpha=alpha)

        return output

    @layer
    def dropout(self, input, rate, name):
        return tf.layers.dropout(input, rate=rate, training=self.is_training, name=name)

    @layer
    def padding(self, input, padding, name, mode='CONSTANT'):
        return tf.pad(input, paddings=[[0,0], [padding, padding], [padding, padding], [0,0]], mode=mode, name=name)

    @layer
    def resize(self, input, factor=None, shape=None, name=None, method=tf.image.ResizeMethod.BILINEAR):
        if factor is not None:
            input_shape = tf.shape(input)
            output_shape = input_shape[1:3] * factor
        elif shape is not None:
            output_shape = shape
        else:
            print('Neither factor nor shape is set for resize layer')
        return tf.image.resize_images(images = input, size = output_shape, method=method)
    @layer
    def leaky_relu(self, input, alpha=0.2, name=None):
        return tf.nn.leaky_relu(input, alpha=alpha, name=name)


