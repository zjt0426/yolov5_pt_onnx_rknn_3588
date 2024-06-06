yolov5_pt-->onnx-->rknn-->3588:
# 将yolov5s.pt转成yolov5s.onnx，将yolov5s.onnx转成yolov5s.rknn，将yolov5s.rknn部署到rk3588并进行推理。
# 1.配置yolov5环境
conda create -n rknn python=3.9
conda activate rknn
进入到requirements.txt目录下。
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
运行以下命令将pt格式转为onnx格式。
python export.py --weights weights/yolov5s.pt --img 640 --batch 1 --include onnx

# 2.配置rknn-toolkit2的onnx转rknn模型环境
cd rknn-toolkit2
pip install -r requirements_cp39-1.6.0.txt -i https://mirror.baidu.com/pypi/simple
pip install rknn_toolkit2-1.6.0+81f21f4d-cp39-cp39-linux_x86_64.whl
验证安装环境：
(rknn)**@**:./rknn-toolkit2$:python
>>>from rknn.api import RKNN
>>>
运行以下命令将onnx转换为rknn模型 
python onnx2rknn.py

# 3.推理验证
以上模型转换需在linux系统中完成，以下是在arm板卡推理流程
安装rknn-toolkit-lite2
进入到yolov5主目录下，运行以下命令
pip install rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl
sudo cp librknnrt.so /usr/lib/

验证自带模型
python test.py

验证自己的模型
python myrknn.py

