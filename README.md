# MXNet_To_ONNX_Tensorflow
mxnet 模型转ONNX 和 tensorflow
## 创建环境
```
conda create -n mxnet python=3.6
conda activate mxnet
pip install mxnet==1.4.0
```

## MXNet 转 ONNX
```
python Mxnet_To_ONNX.py
```

## ONNX 转 tensorflow
需要下载 onnx_tf [下载地址](https://github.com/onnx/onnx-tensorflow)
```
pip install -e ..
```


