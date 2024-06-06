### yolov5_pt-->onnx-->rknn-->3588:
### 本项目主要完成板卡部署推理功能，实现yolov5s.pt转yolov5s.onnx，yolov5s.onnx转yolov5s.rknn，yolov5s.rknn部署到rk3588并进行推理。
## 1.配置yolov5环境
conda create -n rknn python=3.9 <br>
conda activate rknn <br>
进入到requirements.txt目录下。 <br>
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple <br>
运行以下命令将pt格式转为onnx格式。 <br>
python export.py --weights weights/yolov5s.pt --img 640 --batch 1 --include onnx <br>

## 2.配置rknn-toolkit2的onnx转rknn模型环境
cd rknn-toolkit2<br>
pip install -r requirements_cp39-1.6.0.txt -i https://mirror.baidu.com/pypi/simple<br>
pip install rknn_toolkit2-1.6.0+81f21f4d-cp39-cp39-linux_x86_64.whl<br>
验证rknn-toolkit2是否按安装成功：<br>
(rknn)**@**:./rknn-toolkit2$:python<br>
>>>from rknn.api import RKNN<br>
>>><br>
运行以下命令将onnx转换为rknn模型 <br>
python onnx2rknn.py<br>

## 3.推理验证
以上模型转换需在linux系统中完成，以下是在arm板卡推理流程<br>
安装rknn-toolkit-lite2<br>
进入到yolov5主目录下，运行以下命令<br>
pip install rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl<br>
sudo cp librknnrt.so /usr/lib/<br>

验证自带模型<br>
python test.py<br>

验证自己的模型<br>
python myrknn.py<br>

