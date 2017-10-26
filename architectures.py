from __future__ import print_function, division, unicode_literals
import tensorflow as tf
import tensorflow.contrib.slim.nets
import numpy as np
from base import SegmentationModel, PretrainedSegmentationModel


class SimpleSegA(SegmentationModel):
    def __init__(self, name, img_shape, n_channels=3, n_classes=1, dynamic=False, l2=None, best_evals_metric="valid_iou"):
        super().__init__(name=name, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

    def create_body_ops(self):
        """ Ops to make use of:
               self.is_training
               self.X
               self.Y
               self.alpha
               self.dropout
               self.l2_scale
               self.l2
               self.n_classes
        """
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        he_init = tf.contrib.keras.initializers.he_normal()
        relu = tf.nn.relu
        n_classes = self.n_classes
        conv2d = tf.contrib.layers.conv2d
        deconv = tf.contrib.layers.conv2d_transpose

        with tf.contrib.framework.arg_scope(\
            [tf.contrib.layers.conv2d], \
            padding = "SAME",
            stride = 2,
            activation_fn =tf.nn.relu,
            normalizer_fn = None,
            normalizer_params = {"is_training": self.is_training},
            weights_initializer =tf.contrib.layers.xavier_initializer(),
            weights_regularizer =None,
            variables_collections =None,
            trainable =True):

            # DOWN CONVOLUTIONS
            d1 = conv2d(x, num_outputs=8, kernel_size=3, scope="d1")
            print("d1", d1.shape.as_list())
            d2 = conv2d(d1, num_outputs=32, kernel_size=3, scope="d2")
            print("d2", d2.shape.as_list())
            d3 = conv2d(d2, num_outputs=64, kernel_size=3, scope="d3")
            print("d3", d3.shape.as_list())
            d4 = conv2d(d3, num_outputs=64, kernel_size=3, scope="d4")
            print("d4", d4.shape.as_list())

        with tf.contrib.framework.arg_scope([deconv, conv2d], \
            padding = "SAME",
            stride = 2,
            activation_fn = None,
            normalizer_fn = None,
            normalizer_params = {"is_training": self.is_training},
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            weights_regularizer = None,
            variables_collections = None,
            trainable = True):

            # UP CONVOLUTIONS
            with tf.variable_scope('u3') as scope:
                u3 = deconv(d4, num_outputs=n_classes, kernel_size=4, stride=2)
                s3 = conv2d(d3, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="s")
                u3 = tf.add(u3, s3, name="up")
                print("u3", u3.shape.as_list())

            with tf.variable_scope('u2') as scope:
                u2 = deconv(u3, num_outputs=n_classes, kernel_size=4, stride=2)
                s2 = conv2d(d2, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="s")
                u2 = tf.add(u2, s2, name="up")
                print("u2", u2.shape.as_list())

            with tf.variable_scope('u1') as scope:
                u1 = deconv(u2, num_outputs=n_classes, kernel_size=4, stride=2)
                s1 = conv2d(d1, num_outputs=n_classes, kernel_size=1, stride=1, activation_fn=relu, scope="s")
                u1 = tf.add(u1, s1, name="up")
                print("u1", u1.shape.as_list())

            self.logits = deconv(u1, num_outputs=n_classes, kernel_size=4, stride=2, activation_fn=None, scope="logits")


class InceptionV3_SegmenterA(PretrainedSegmentationModel):
    def __init__(self, name, pretrained_snapshot, img_shape=299, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_iou"):
        super().__init__(name=name, pretrained_snapshot=pretrained_snapshot, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

    def create_body_ops(self):
        """ Inception V3  model
            Dropout before skip connections
            Uses rescaled inputs (0-1)
            Contains Bathchnorm in layers

            DOWNSAMLIGN HALF
            - pretrained Ineption v3, 229x299 images

            UPSAMPLING HALF
            - Skip layers use conv 1x1
            - Add(upsample-> batchnorm > relu,  skip>batchnorm>relu)
        """
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        is_training = self.is_training
        winit = tf.contrib.layers.xavier_initializer()
        relu = tf.nn.relu
        n_classes = self.n_classes
        conv2d = tf.contrib.layers.conv2d
        deconv = tf.contrib.layers.conv2d_transpose
        batchnorm = tf.contrib.layers.batch_norm
        drop = 0.5

        # INCEPTION TRUNK
        arg_scope = tf.contrib.slim.nets.inception.inception_v3_arg_scope
        # architecture_func = tf.contrib.slim.nets.inception.inception_v3_base
        with tf.contrib.framework.arg_scope(arg_scope()):
            with tf.variable_scope("InceptionV3", 'InceptionV3') as scope:
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):
                    _, end_points = tf.contrib.slim.nets.inception.inception_v3_base(
                          inputs=x,
                          final_endpoint='Mixed_7c',
                          scope=scope)

        for k,v in end_points.items():
            print(k, v.shape.as_list())

        # IMPORTANT DOWN SAMPLE LAYERS
        ops = {}
        ops["d1"] = end_points["Conv2d_2b_3x3"]
        ops["d2"] = end_points["Conv2d_4a_3x3"]
        ops["d3"] = end_points["Mixed_5d"]
        ops["d4"] = end_points["Mixed_6e"]
        ops["d5"] = end_points["Mixed_7c"]

        # ADD DROPOUT LAYERS
        with tf.variable_scope('d1') as scope:
            x = ops["d1"]
            d1 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d2') as scope:
            x = ops["d2"]
            d2 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d3') as scope:
            x = ops["d3"]
            d3 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d4') as scope:
            x = ops["d4"]
            d4 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d5') as scope:
            x = ops["d5"]
            d5 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        # UPSAMPLING
        with tf.variable_scope("upsampling"):
            with tf.contrib.framework.arg_scope([deconv, conv2d], \
                padding = "VALID",
                stride = 1,
                activation_fn = relu,
                normalizer_fn = batchnorm,
                # normalizer_params = {"is_training": self.is_training},
                weights_initializer = winit,
                weights_regularizer = None,
                variables_collections = None,
                trainable = True
                ):
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):

                    print("d5", d5.shape.as_list())
                    print("d4", d4.shape.as_list())

                    # UP CONVOLUTIONS
                    with tf.variable_scope('u4') as scope:
                        previous, residual = d5, d4
                        k, stride = 3, 2
                        u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                        s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                        u4 = tf.add(u, s, name="up")
                        print("u4", u4.shape.as_list())

                    with tf.variable_scope('u3') as scope:
                        previous, residual = u4, d3
                        k, stride = 3, 2
                        u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                        s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                        u3 = tf.add(u, s, name="up")
                        print("d3", d3.shape.as_list())
                        print("u3", u3.shape.as_list())

                    with tf.variable_scope('u2') as scope:
                        previous, residual = u3, d2
                        k, stride = 3, 2
                        u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                        s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                        u2 = tf.add(u, s, name="up")
                        print("d2", d2.shape.as_list())
                        print("u2", u2.shape.as_list())

                    with tf.variable_scope('u1') as scope:
                        previous, residual = u2, d1
                        k, stride = 7, 2
                        u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                        s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                        u1 = tf.add(u, s, name="up")
                        print("d1", d1.shape.as_list())
                        print("u1", u1.shape.as_list())

                    with tf.variable_scope('logits') as scope:
                        previous = u1
                        k, stride = 7, 2
                        self.logits = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride, normalizer_fn=None, activation_fn=None)
                        print("logits", self.logits.shape.as_list())


class InceptionV3_SegmenterB(PretrainedSegmentationModel):
    def __init__(self, name, pretrained_snapshot, img_shape=299, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_iou"):
        super().__init__(name=name, pretrained_snapshot=pretrained_snapshot, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

    def create_body_ops(self):
        """ Inception V3  model
            Dropout before skip connections
            Uses rescaled inputs (0-1)
            Contains Bathchnorm in layers
            Relu only after adition of skip connections

            DOWNSAMLIGN HALF
            - pretrained Ineption v3, 229x299 images

            UPSAMPLING HALF
            - Skip layers use conv 1x1
            - Add(upsample-> batchnorm,  skip>batchnorm) -> Relu
        """
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        is_training = self.is_training
        winit = tf.contrib.layers.xavier_initializer()
        relu = tf.nn.relu
        n_classes = self.n_classes
        conv2d = tf.contrib.layers.conv2d
        deconv = tf.contrib.layers.conv2d_transpose
        batchnorm = tf.contrib.layers.batch_norm
        drop = 0.5

        # INCEPTION TRUNK
        arg_scope = tf.contrib.slim.nets.inception.inception_v3_arg_scope
        # architecture_func = tf.contrib.slim.nets.inception.inception_v3_base
        with tf.contrib.framework.arg_scope(arg_scope()):
            with tf.variable_scope("InceptionV3", 'InceptionV3') as scope:
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):
                    _, end_points = tf.contrib.slim.nets.inception.inception_v3_base(
                          inputs=x,
                          final_endpoint='Mixed_7c',
                          scope=scope)

        for k,v in end_points.items():
            print(k, v.shape.as_list())

        # IMPORTANT DOWN SAMPLE LAYERS
        ops = {}
        ops["d1"] = end_points["Conv2d_2b_3x3"]
        ops["d2"] = end_points["Conv2d_4a_3x3"]
        ops["d3"] = end_points["Mixed_5d"]
        ops["d4"] = end_points["Mixed_6e"]
        ops["d5"] = end_points["Mixed_7c"]

        # ADD DROPOUT LAYERS
        with tf.variable_scope('d1') as scope:
            x = ops["d1"]
            d1 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d2') as scope:
            x = ops["d2"]
            d2 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d3') as scope:
            x = ops["d3"]
            d3 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d4') as scope:
            x = ops["d4"]
            d4 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d5') as scope:
            x = ops["d5"]
            d5 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        # UPSAMPLING
        with tf.variable_scope("upsampling"):
            with tf.contrib.framework.arg_scope([deconv, conv2d], \
                padding = "VALID",
                stride = 1,
                activation_fn = None,
                normalizer_fn = batchnorm,
                normalizer_params = {"is_training": self.is_training},
                weights_initializer = winit,
                weights_regularizer = None,
                variables_collections = None,
                trainable = True
                ):
                # with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):

                print("d5", d5.shape.as_list())
                print("d4", d4.shape.as_list())

                # UP CONVOLUTIONS
                with tf.variable_scope('u4') as scope:
                    previous, residual = d5, d4
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u4 = tf.add(u, s, name="up")
                    u4 = relu(u4)
                    print("u4", u4.shape.as_list())

                with tf.variable_scope('u3') as scope:
                    previous, residual = u4, d3
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u3 = tf.add(u, s, name="up")
                    u3 = relu(u3)
                    print("d3", d3.shape.as_list())
                    print("u3", u3.shape.as_list())

                with tf.variable_scope('u2') as scope:
                    previous, residual = u3, d2
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u2 = tf.add(u, s, name="up")
                    u2 = relu(u2)
                    print("d2", d2.shape.as_list())
                    print("u2", u2.shape.as_list())

                with tf.variable_scope('u1') as scope:
                    previous, residual = u2, d1
                    k, stride = 7, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u1 = tf.add(u, s, name="up")
                    u1 = relu(u1)
                    print("d1", d1.shape.as_list())
                    print("u1", u1.shape.as_list())

                with tf.variable_scope('logits') as scope:
                    previous = u1
                    k, stride = 7, 2
                    self.logits = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride, normalizer_fn=None, activation_fn=None)
                    print("logits", self.logits.shape.as_list())




class InceptionV3_SegmenterC(PretrainedSegmentationModel):
    def __init__(self, name, pretrained_snapshot, img_shape=299, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_iou"):
        super().__init__(name=name, pretrained_snapshot=pretrained_snapshot, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

    def create_body_ops(self):
        """ Inception V3  model
            Dropout before skip connections
            Uses rescaled inputs (0-1)
            Contains Bathchnorm in layers
            NO Relu in up sample layers

            DOWNSAMLIGN HALF
            - pretrained Ineption v3, 229x299 images

            UPSAMPLING HALF
            - Skip layers use conv 1x1
            - Add(upsample-> batchnorm,  skip>batchnorm)
        """
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        is_training = self.is_training
        winit = tf.contrib.layers.xavier_initializer()
        relu = tf.nn.relu
        n_classes = self.n_classes
        conv2d = tf.contrib.layers.conv2d
        deconv = tf.contrib.layers.conv2d_transpose
        batchnorm = tf.contrib.layers.batch_norm
        drop = 0.5

        # INCEPTION TRUNK
        arg_scope = tf.contrib.slim.nets.inception.inception_v3_arg_scope
        # architecture_func = tf.contrib.slim.nets.inception.inception_v3_base
        with tf.contrib.framework.arg_scope(arg_scope()):
            with tf.variable_scope("InceptionV3", 'InceptionV3') as scope:
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):
                    _, end_points = tf.contrib.slim.nets.inception.inception_v3_base(
                          inputs=x,
                          final_endpoint='Mixed_7c',
                          scope=scope)

        for k,v in end_points.items():
            print(k, v.shape.as_list())

        # IMPORTANT DOWN SAMPLE LAYERS
        ops = {}
        ops["d1"] = end_points["Conv2d_2b_3x3"]
        ops["d2"] = end_points["Conv2d_4a_3x3"]
        ops["d3"] = end_points["Mixed_5d"]
        ops["d4"] = end_points["Mixed_6e"]
        ops["d5"] = end_points["Mixed_7c"]

        # ADD DROPOUT LAYERS
        with tf.variable_scope('d1') as scope:
            x = ops["d1"]
            d1 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d2') as scope:
            x = ops["d2"]
            d2 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d3') as scope:
            x = ops["d3"]
            d3 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d4') as scope:
            x = ops["d4"]
            d4 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d5') as scope:
            x = ops["d5"]
            d5 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        # UPSAMPLING
        with tf.variable_scope("upsampling"):
            with tf.contrib.framework.arg_scope([deconv, conv2d], \
                padding = "VALID",
                stride = 1,
                activation_fn = None,
                normalizer_fn = batchnorm,
                normalizer_params = {"is_training": self.is_training},
                weights_initializer = winit,
                weights_regularizer = None,
                variables_collections = None,
                trainable = True
                ):
                # with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):

                print("d5", d5.shape.as_list())
                print("d4", d4.shape.as_list())

                # UP CONVOLUTIONS
                with tf.variable_scope('u4') as scope:
                    previous, residual = d5, d4
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u4 = tf.add(u, s, name="up")
                    print("u4", u4.shape.as_list())

                with tf.variable_scope('u3') as scope:
                    previous, residual = u4, d3
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u3 = tf.add(u, s, name="up")
                    print("d3", d3.shape.as_list())
                    print("u3", u3.shape.as_list())

                with tf.variable_scope('u2') as scope:
                    previous, residual = u3, d2
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u2 = tf.add(u, s, name="up")
                    print("d2", d2.shape.as_list())
                    print("u2", u2.shape.as_list())

                with tf.variable_scope('u1') as scope:
                    previous, residual = u2, d1
                    k, stride = 7, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u1 = tf.add(u, s, name="up")
                    print("d1", d1.shape.as_list())
                    print("u1", u1.shape.as_list())

                with tf.variable_scope('logits') as scope:
                    previous = u1
                    k, stride = 7, 2
                    self.logits = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride, normalizer_fn=None, activation_fn=None)
                    print("logits", self.logits.shape.as_list())



class InceptionV3_SegmenterD(PretrainedSegmentationModel):
    def __init__(self, name, pretrained_snapshot, img_shape=299, n_channels=3, n_classes=10, dynamic=False, l2=None, best_evals_metric="valid_iou"):
        super().__init__(name=name, pretrained_snapshot=pretrained_snapshot, img_shape=img_shape, n_channels=n_channels, n_classes=n_classes, dynamic=dynamic, l2=l2, best_evals_metric=best_evals_metric)

    def create_body_ops(self):
        """ Inception V3  model
            Dropout before skip connections
            Uses rescaled inputs (0-1)
            No Bathchnorm in layers upsampling layers
            NO Relu in up sample layers

            DOWNSAMLIGN HALF
            - pretrained Ineption v3, 229x299 images

            UPSAMPLING HALF
            - Skip layers use conv 1x1
            - Add(upsample,  skip)
        """
        with tf.name_scope("preprocess") as scope:
            x = tf.div(self.X, 255, name="rescaled_inputs")

        is_training = self.is_training
        winit = tf.contrib.layers.xavier_initializer()
        relu = tf.nn.relu
        n_classes = self.n_classes
        conv2d = tf.contrib.layers.conv2d
        deconv = tf.contrib.layers.conv2d_transpose
        batchnorm = tf.contrib.layers.batch_norm
        drop = 0.5

        # INCEPTION TRUNK
        arg_scope = tf.contrib.slim.nets.inception.inception_v3_arg_scope
        # architecture_func = tf.contrib.slim.nets.inception.inception_v3_base
        with tf.contrib.framework.arg_scope(arg_scope()):
            with tf.variable_scope("InceptionV3", 'InceptionV3') as scope:
                with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):
                    _, end_points = tf.contrib.slim.nets.inception.inception_v3_base(
                          inputs=x,
                          final_endpoint='Mixed_7c',
                          scope=scope)

        for k,v in end_points.items():
            print(k, v.shape.as_list())

        # IMPORTANT DOWN SAMPLE LAYERS
        ops = {}
        ops["d1"] = end_points["Conv2d_2b_3x3"]
        ops["d2"] = end_points["Conv2d_4a_3x3"]
        ops["d3"] = end_points["Mixed_5d"]
        ops["d4"] = end_points["Mixed_6e"]
        ops["d5"] = end_points["Mixed_7c"]

        # ADD DROPOUT LAYERS
        with tf.variable_scope('d1') as scope:
            x = ops["d1"]
            d1 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d2') as scope:
            x = ops["d2"]
            d2 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d3') as scope:
            x = ops["d3"]
            d3 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d4') as scope:
            x = ops["d4"]
            d4 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        with tf.variable_scope('d5') as scope:
            x = ops["d5"]
            d5 = tf.layers.dropout(x, rate=drop, training=is_training, name="dropout")

        # UPSAMPLING
        with tf.variable_scope("upsampling"):
            with tf.contrib.framework.arg_scope([deconv, conv2d], \
                padding = "VALID",
                stride = 1,
                activation_fn = None,
                normalizer_fn = None,
                normalizer_params = {"is_training": self.is_training},
                weights_initializer = winit,
                weights_regularizer = None,
                variables_collections = None,
                trainable = True
                ):
                # with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm, tf.contrib.layers.dropout], is_training=self.is_training):

                print("d5", d5.shape.as_list())
                print("d4", d4.shape.as_list())

                # UP CONVOLUTIONS
                with tf.variable_scope('u4') as scope:
                    previous, residual = d5, d4
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u4 = tf.add(u, s, name="up")
                    print("u4", u4.shape.as_list())

                with tf.variable_scope('u3') as scope:
                    previous, residual = u4, d3
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u3 = tf.add(u, s, name="up")
                    print("d3", d3.shape.as_list())
                    print("u3", u3.shape.as_list())

                with tf.variable_scope('u2') as scope:
                    previous, residual = u3, d2
                    k, stride = 3, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u2 = tf.add(u, s, name="up")
                    print("d2", d2.shape.as_list())
                    print("u2", u2.shape.as_list())

                with tf.variable_scope('u1') as scope:
                    previous, residual = u2, d1
                    k, stride = 7, 2
                    u = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride)
                    s = conv2d(residual, num_outputs=n_classes, kernel_size=1, stride=1, scope="s")
                    u1 = tf.add(u, s, name="up")
                    print("d1", d1.shape.as_list())
                    print("u1", u1.shape.as_list())

                with tf.variable_scope('logits') as scope:
                    previous = u1
                    k, stride = 7, 2
                    self.logits = deconv(previous, num_outputs=n_classes, kernel_size=k, stride=stride, normalizer_fn=None, activation_fn=None)
                    print("logits", self.logits.shape.as_list())


arc = {}
arc["SimpleSegA"] = SimpleSegA
arc["InceptionV3_SegmenterA"] = InceptionV3_SegmenterA
arc["InceptionV3_SegmenterB"] = InceptionV3_SegmenterB
arc["InceptionV3_SegmenterC"] = InceptionV3_SegmenterC
arc["InceptionV3_SegmenterD"] = InceptionV3_SegmenterD
