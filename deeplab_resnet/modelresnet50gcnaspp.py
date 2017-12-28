# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf


class DeepLabResNetModelOri50gcnaspp(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
         .conv(7, 7, 64, 2, 2, biased=True, relu=False, name='conv1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add(name='res2c')
         .relu(name='res2c_relu')
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
         .add(name='res3b')
         .relu(name='res3b_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3c_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3c_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3c_branch2c'))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
         .add(name='res3c')
         .relu(name='res3c_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3d_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3d_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3d_branch2c'))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
         .add(name='res3d')
         .relu(name='res3d_relu')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3d_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4c_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4c_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4c_branch2c'))

        (self.feed('res4b_relu',
                   'bn4c_branch2c')
         .add(name='res4c')
         .relu(name='res4c_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4d_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4d_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4d_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4d_branch2c'))

        (self.feed('res4c_relu',
                   'bn4d_branch2c')
         .add(name='res4d')
         .relu(name='res4d_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4e_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4e_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4e_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4e_branch2c'))

        (self.feed('res4d_relu',
                   'bn4e_branch2c')
         .add(name='res4e')
         .relu(name='res4e_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4f_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4f_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4f_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4f_branch2c'))

        (self.feed('res4e_relu',
                   'bn4f_branch2c')
         .add(name='res4f')
         .relu(name='res4f_relu')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4f_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
         .add(name='res5a')
         .relu(name='res5a_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add(name='res5b')
         .relu(name='res5b_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
         .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add(name='res5c')
         .relu(name='res5c_relu')
         .conv(15, 1, num_classes, 1, 1, biased=False, relu=False, name='gcn1_branch1a')
         .conv(1, 15, num_classes, 1, 1, biased=False, relu=False, name='gcn1_branch1b'))

        (self.feed('res5c_relu')
         .conv(1, 15, num_classes, 1, 1, biased=False, relu=False, name='gcn1_branch2a')
         .conv(15, 1, num_classes, 1, 1, biased=False, relu=False, name='gcn1_branch2b'))

        (self.feed('gcn1_branch1b',
                   'gcn1_branch2b')
         .add(name='gcn1')
         .relu(name='gcn1_relu')
         .conv(3, 3, num_classes, 1, 1, biased=False, relu=True, name='br1_branch1a')
         .conv(3, 3, num_classes, 1, 1, biased=False, relu=False, name='br1_branch1b'))

        (self.feed('gcn1_relu',
                   'br1_branch1b')
         .add(name='br1')
         .relu(name='br1_relu'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

        (self.feed('br1_relu',
                   'fc1_voc12_c0',
                   'fc1_voc12_c1',
                   'fc1_voc12_c2',
                   'fc1_voc12_c3')
         .add(name='fc1_voc12'))

