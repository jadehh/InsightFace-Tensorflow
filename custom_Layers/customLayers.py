#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：2019/8/26 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/26  下午8:21 modify by jade
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
def l2_normalize_caffe(x,epsilon=1e-10,pow=2,dim=None,keepdims=None,name=None):
    with ops.name_scope(name, "l2_normalize_caffe", [x]) as name:
      x = ops.convert_to_tensor(x, name="x")
      x_pow = math_ops.pow(x, pow)
      x_sum = math_ops.reduce_sum(x_pow,[dim],keepdims=True)
      x_sqrt = math_ops.sqrt(x_sum)
      x = math_ops.add(x_sqrt,epsilon)
      return x
