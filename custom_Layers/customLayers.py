#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/26  下午8:21 modify by jade
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python import keras
from tensorflow.python.keras import initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import backend as K
def l2_normalize_caffe(x, epsilon=1e-10, pow=2, dim=None, keepdims=None, name=None):
    with ops.name_scope(name, "l2_normalize_caffe", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        x_pow = math_ops.pow(x, pow)
        x_sum = math_ops.reduce_sum(x_pow, [dim], keepdims=True)
        x_sqrt = math_ops.sqrt(x_sum)
        x = math_ops.add(x_sqrt, epsilon)
        return x


class BatchNorm2D(Layer):
    def __init__(self, out_c, epsilon=1e-10, name=''):
        super(BatchNorm2D, self).__init__()
        self.output_dim = out_c
        self.epsilon = epsilon
        # kernel_initializer = 'glorot_uniform',
        # self.kernel_initializer = initializers.get(kernel_initializer)

        self.weight = self.add_weight(shape=[out_c],initializer=keras.initializers.random_normal,trainable=True,name=name+"_weights")
        # w_init = tf.random_normal_initializer()
        # self.weight = tf.Variable(initial_value=w_init(
        #     shape=[out_c], dtype=tf.float32), trainable=True,name=name+"_weights")


    def call(self, inputs):
        norm = l2_normalize_caffe(x=inputs,
                                  epsilon=self.epsilon,
                                  pow=2,
                                  dim=3,
                                  keepdims=True,
                                  name=self.name + "_batchnorm")
        inputs /= norm
        weights = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.weight, 0), 0), 0)
        x1_expand = tf.tile(weights, [1, inputs.get_shape()[2], inputs.get_shape()[2], 1])
        inputs = x1_expand * inputs
        return inputs


class GroupConv(Layer):
    """
    分组卷积,将各个通道分离出来在各自卷积
    """
    def __init__(self, out_c, kernel=(1, 1), strides=(1, 1), padding=(0, 0),groups=1,name="",bias=True):
        super(GroupConv, self).__init__()
        self.out_c = out_c
        self.kernel_size = kernel
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.layer_name = name
        self.bias = bias
    def build(self, input_shape):
        if self.padding is None:
            self.pad = 'same'
        else:
            self.zero_pad =ZeroPadding2D(padding=(self.padding, self.padding), name=self.layer_name + "_padding")
            self.pad = 'valid'
        if self.groups == 1 or self.groups is None:
                #这个就是普通的卷积
            if self.bias:
                self.conv = Conv2D(self.out_c,kernel_size=self.kernel_size,strides=self.strides,activation='relu',padding=self.pad,name=self.layer_name)
            else:
                self.conv = Conv2D(self.out_c,kernel_size=self.kernel_size,strides=self.strides,activation='relu',padding=self.pad,name=self.layer_name,use_bias=False)
        else:
            #分组卷积
            self.group_convs = []
            for i in range(self.groups):
                if self.bias:
                    self.group_convs.append(Conv2D(int(self.out_c / self.groups) ,kernel_size=self.kernel_size,strides=self.strides,activation='relu',padding=self.pad,name=self.layer_name+"_"+str(i)))
                else:
                    self.group_convs.append(Conv2D(int(self.out_c / self.groups) ,kernel_size=self.kernel_size,strides=self.strides,activation='relu',padding=self.pad,name=self.layer_name+"_"+str(i),use_bias=False))
    def call(self, x):
        if self.padding is not None:
            x = self.zero_pad(x)
        if self.groups == 1 or self.groups is None:
            x = self.conv(x)
            return x
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=self.groups, value=x)
            output_groups = []
            for i in range(self.groups):
                output_group = self.group_convs[i](input_groups[i])
                output_groups.append(output_group)
            return tf.concat(axis=3, values=output_groups)




class Conv_block(Layer):
    def __init__(self,out_c,kernel=(1,1),stride=(1,1),padding=None,groups=None,name=''):
        super(Conv_block,self).__init__()
            # self.conv = SeparableConv2D
        self.conv = GroupConv(out_c, kernel=kernel, strides=stride, padding=padding, groups=groups,name=name + "_conv")
        self.bn = BatchNorm2D(out_c,name=name+"_bn")
        self.prelu = PReLU(name=name+"_prelu")


    def call(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x



class Linear_block(Layer):
    def __init__(self, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1,name=""):
        super(Linear_block, self).__init__()
        self.conv = GroupConv(out_c,kernel=kernel,strides=stride,padding=padding,groups=groups,name=name+"_liner",bias=False)
        self.bn = BatchNorm2D(out_c,name=name+"_bn")
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Layer):
    def __init__(self,out_c, residual = False,kernel=(3, 3), stride=(2, 2), padding=(1, 1),groups=1,name=""):
        super(Depth_Wise, self).__init__()
        self.out_c = out_c
        self.kernel_size = kernel
        self.strides = stride
        self.padding = padding
        self.groups = groups
        self.layer_name = name
        self.residual = residual
    def build(self, input_shape):
        self.conv = Conv_block(self.groups, kernel=(1,1), padding=None, stride=(1, 1),name=self.layer_name)
        self.conv_dw = Conv_block(self.groups, kernel=self.kernel_size, padding=self.padding, stride=self.strides,groups=self.groups,name=self.layer_name+"_dw")
        self.project = Linear_block(self.out_c, kernel=(1,1), padding=None, stride=(1, 1),name=self.layer_name)

    def call(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Layer):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1),name=""):
        super(Residual, self).__init__()
        self.c = c
        self.num_block = num_block
        self.groups = groups
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.layer_name = name

    def build(self, input_shape):
        modules = []
        for i in range(self.num_block):
            modules.append(Depth_Wise(self.c, residual=True, kernel=self.kernel, padding=self.padding, stride=self.stride, groups=self.groups,name=self.layer_name + "_" + str(i)))
        self.models = modules

    def call(self, x):
        for model in self.models:
            x = model(x)
        return x
