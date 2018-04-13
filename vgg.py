import tensorflow as tf
import numpy as np

class VGG19(object):
    def __init__(self, num_channels_in, sess=None):
        if sess == None:
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run() 
        else:
            self.sess = sess
        self._build_model(num_channels_in, 224, 224)

    def _build_model(self, num_channels_in, im_height, im_width):
        with tf.variable_scope('vgg'):
            self.im_input = tf.placeholder(tf.float32, [None, num_channels_in, im_height, im_width])

            net = self._conv_layer3(self.im_input, 64, 'conv1')
            net = self._conv_layer3(net, 64, 'conv2')
            net = self._max_pool(net, 'pool1')

            net = self._conv_layer3(net, 128, 'conv3')
            net = self._conv_layer3(net, 128, 'conv4')
            net = self._max_pool(net, 'pool2')

            net = self._conv_layer3(net, 256, 'conv5')
            net = self._conv_layer3(net, 256, 'conv6')
            net = self._conv_layer3(net, 256, 'conv7')
            net = self._conv_layer3(net, 256, 'conv8')
            net = self._max_pool(net, 'pool3')

            net = self._conv_layer3(net, 512, 'conv9')
            net = self._conv_layer3(net, 512, 'conv10')
            net = self._conv_layer3(net, 512, 'conv11')
            net = self._conv_layer3(net, 512, 'conv12')
            net = self._max_pool(net, 'pool4')

            net = self._conv_layer3(net, 512, 'conv13')
            net = self._conv_layer3(net, 512, 'conv14')
            net = self._conv_layer3(net, 512, 'conv15')
            net = self._conv_layer3(net, 512, 'conv16')
            net = self._max_pool(net, 'pool5')

            net = self._reshape_to_linear(net)
            net = self._fc_layer(net, 4096, 'fc1')
            net = self._fc_layer(net, 4096, 'fc2')
            net = self._fc_layer(net, 1000, 'fc3')

            self.output = tf.nn.softmax(net, name='output')

            # training
            loss = None

            self.lr = tf.placeholder(tf.float32, ())
            tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)


    def _reshape_to_linear(self, input):
        shape = input.get_shape().as_list()
        feature_num = np.prod(shape[1:])
        return tf.reshape(input, [-1, feature_num])


    def _conv_layer3(self, input, num_outputs, name):
        layer = tf.contrib.layers.conv2d(input,
            num_outputs=num_outputs,
            kernel_size=3,
            stride=1,
            padding='SAME',
            data_format='NCHW',
            activation_fn=tf.nn.relu,
            scope=name)
        return layer

    def _conv_layer1(self, input, num_outputs, name):
        layer = tf.contrib.layers.conv2d(input,
            num_outputs=num_outputs,
            kernel_size=1,
            stride=1,
            padding='SAME',
            data_format='NCHW',
            activation_fn=tf.nn.relu,
            scope=name)
        return layer

    def _max_pool(self, input, name):
        layer = tf.nn.max_pool(input,
            ksize=[1, 1, 2, 2],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NCHW',
            name=name)
        # layer = tf.contrib.max_pool2d(input, 
        #     kernel_size=2,
        #     stride=2,
        #     padding='SAME',
        #     data_format='NCHW',
        #     scope=name)
        return layer

    def _fc_layer(self, input, num_outputs, name, activation=tf.nn.relu):
        layer = tf.contrib.layers.fully_connected(input,
            num_outputs=num_outputs,
            activation_fn=activation,
            scope=name)
        return layer


if __name__ == '__main__':
    vgg = VGG19(2)
    data = np.load('data/vgg19.npy', encoding='latin1').item()
    import pdb
    pdb.set_trace()


