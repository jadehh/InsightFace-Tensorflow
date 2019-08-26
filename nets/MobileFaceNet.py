#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/26  下午8:21 modify by jade

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model
import numpy as np
from custom_Layers.customLayers import *
import cv2
##############################  MobileFaceNet ############################

class BatchNorm2D(Layer):
    def __init__(self,out_c,epsilon=1e-10,name=''):
        super(BatchNorm2D, self).__init__()
        self.output_dim = out_c
        self.epsilon = epsilon
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=[out_c], dtype=tf.float32), trainable=True)



    def call(self,inputs):
        norm = l2_normalize_caffe(x=inputs,
                               epsilon=self.epsilon,
                               pow=2,
                               dim=3,
                               keepdims=True,
                               name=self.name+"_batchnorm")
        inputs /= norm
        weights= tf.expand_dims(tf.expand_dims(tf.expand_dims(self.weight, 0), 0), 0)
        x1_expand = tf.tile(weights, [1, inputs.get_shape()[2], inputs.get_shape()[2], 1])
        inputs = x1_expand * inputs
        return  inputs


class Conv_block(Model):
    def __init__(self,out_c,kernel=(1,1),stride=(1,1),padding=(0,0),name=''):
        super(Conv_block,self).__init__()
        self.pad = ZeroPadding2D(padding=(padding,padding),name=name+"_pad")
        self.conv = Conv2D(out_c,kernel_size=kernel,strides=stride,padding="valid",name=name+"_conv")
        self.bn = BatchNorm2D(out_c,name=name)
        self.prelu = PReLU()


    def call(self,x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        # x = self.prelu(x)
        return x

class MobileFaceNet(Model):
    def __init__(self):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(64,kernel=(3,3),stride=(2,2),padding=(1,1),name="conv1")

    def call(self,x):
        x = self.conv1(x)
        return x



if __name__ == '__main__':
    model = MobileFaceNet()
    model.build(input_shape=(None, 224, 224, 3))
    img = cv2.imread(r"C:\Users\jade\PycharmProjects\InsightFace-Tensorflow\examples/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    x = model.predict(np.array([img]))
    print(x)
