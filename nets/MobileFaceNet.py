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

class MobileFaceNet(Model):
    def __init__(self):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(64,kernel=(3,3),stride=(2,2),padding=(1,1),name="conv1")
        self.conv2_dw = Conv_block(64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64,name="conv2_dw")
        self.conv_23 = Depth_Wise(64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128,name="conv_23")
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1),name="conv_3")
        # self.conv_34 = Depth_Wise(128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256,name="conv_34")
        # self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1),name="conv_4")
        # self.conv_45 = Depth_Wise(128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512,name="conv4_5")

        # self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1),name="conv_5")
        # self.conv_6_sep = Conv_block(512, kernel=(1, 1), stride=(1, 1), padding=(0, 0),name="conv_6_sep")
        # self.conv_6_dw = Linear_block(512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0),name="conv6_6_dw")

        #self.conv_6_flatten = Flatten()









    def call(self,x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        # x = self.conv_34(x)
        # x = self.conv_4(x)
        # x = self.conv_45(x)
        # x = self.conv_5(x)
        # x = self.conv_6_sep(x)
        # x = self.conv_6_dw(x)
        #x = self.conv_6_flatten(x)
        return x



if __name__ == '__main__':
    model = MobileFaceNet()
    model.build(input_shape=(1, 112, 112, 3))
    img = cv2.imread(r"C:\Users\jade\PycharmProjects\InsightFace-Tensorflow\examples/test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (112, 112))
    img = img.astype("float32")
    x = model.predict(np.array([img]))
    print(x)
