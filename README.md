yolov5_pt_onnx_rknn_3588------>将yolov5s.pt转成yolov5s.onnx，将yolov5s.onnx转成yolov5s.rknn，将yolov5s.rknn部署到rk3588并进行推理。
配置yolov5环境：
conda create -n rknn python=3.9
conda activate rknn
进入到requirements.txt目录下。
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
运行以下命令将pt格式转为onnx格式。
python export.py --weights weights/yolov5s.pt --img 640 --batch 1 --include onnx

配置rknn-toolkit2的onnx转rknn模型环境
cd rknn-toolkit2
pip install -r requirements_cp39-1.6.0.txt -i https://mirror.baidu.com/pypi/simple
pip install rknn_toolkit2-1.6.0+81f21f4d-cp39-cp39-linux_x86_64.whl
验证安装环境：
(rknn)**@**:./rknn-toolkit2$:python
>>>from rknn.api import RKNN
>>>
运行以下命令将onnx转换为rknn模型 
python onnx2tknn.py



# 1.训练模型pt格式转onnx

# 2.onnx转rknn

rknn-toolkit2安装

rknn-toolkit-lite2安装（arm架构推理）

rknpu2安装

