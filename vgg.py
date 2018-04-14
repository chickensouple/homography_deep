import tensorflow as tf
import numpy as np
from spatial_transformer import spatial_transformer_network



class VGG19(object):
    def __init__(self, num_channels_in, sess=None):
        self.data_format = 'NHWC'
        self._build_model(num_channels_in, 224, 224)


        if sess == None:
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run() 
        else:
            self.sess = sess


    def _build_model(self, num_channels_in, im_height, im_width):
        with tf.variable_scope('vgg'):
            if self.data_format == 'NCHW':
                self.im_input = tf.placeholder(tf.float32, [None, num_channels_in, im_height, im_width])
            else:
                self.im_input = tf.placeholder(tf.float32, [None, im_height, im_width, num_channels_in])


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

            # affine variables
            self.params = self._fc_layer(net, 6, 'fc3', activation=None,
                biases_initializer=tf.constant_initializer([1, 0, 0, 0, 1, 0]))

            # spatial transformer
            first_image_batch = self.im_input[:, :, :, 0:1]
            second_image_batch = self.im_input[:, :, :, 1:2]
            self.transformed_image_batch = spatial_transformer_network(first_image_batch, self.params)

            # training
            # self.loss = tf.losses.absolute_difference(second_image_batch, self.transformed_image_batch)
            self.loss = tf.nn.l2_loss(second_image_batch - self.transformed_image_batch)


            # self.lr = tf.placeholder(tf.float32, ())
            self.lr = 1e-5
            # self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss)
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def train(self, image_data):
        fd = {self.im_input: image_data}
        loss, _, params, transformed_images = self.sess.run([self.loss, self.optimize, self.params, self.transformed_image_batch], feed_dict=fd)
        return loss, params, transformed_images

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
            data_format=self.data_format,
            activation_fn=tf.nn.relu,
            scope=name)
        return layer

    def _conv_layer1(self, input, num_outputs, name):
        layer = tf.contrib.layers.conv2d(input,
            num_outputs=num_outputs,
            kernel_size=1,
            stride=1,
            padding='SAME',
            data_format=self.data_format,
            activation_fn=tf.nn.relu,
            scope=name)
        return layer

    def _max_pool(self, input, name):
        layer = tf.nn.max_pool(input,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            data_format=self.data_format,
            name=name)
        return layer

    def _fc_layer(self, input, num_outputs, name, activation=tf.nn.relu, **kwargs):
        layer = tf.contrib.layers.fully_connected(input,
            num_outputs=num_outputs,
            activation_fn=activation,
            scope=name,
            **kwargs)
        return layer


if __name__ == '__main__':
    vgg = VGG19(2)
    data = np.load('data/vgg19.npy', encoding='latin1').item()
    import pdb
    pdb.set_trace()


