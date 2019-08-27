#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/27 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/27  上午9:30 modify by jade

import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)

sym = '/home/jade/Models/FaceRecognizeModels/model-y1/model-y1-test2/model-symbol.json'
params = '/home/jade/Models/FaceRecognizeModels/model-y1/model-y1-test2/model-0000.params'

# 标准Imagenet输入- 3通道，224*224
input_shape = (1,3,112,112)

# 输出文件的路径
onnx_file = '/home/jade/Models/FaceRecognizeModels/mxnet_exported_resnet50.onnx'
converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)


