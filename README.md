### yolov5_pt-->onnx-->rknn-->3588:<br>
## 本项目主要完成板卡部署推理功能，实现yolov5s.pt转yolov5s.onnx，yolov5s.onnx转yolov5s.rknn，yolov5s.rknn部署到rk3588并进行推理。<br>
## 1.配置yolov5环境<br>
如需板卡安装anaconda3，则运行sudo wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-aarch64.sh 命令，此处忽略安装过程。<br>
conda create -n rknn python=3.9<br>
conda activate rknn<br>
进入yolov5文件目录下<br>
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple<br>
运行以下命令将pt格式转为onnx格式<br>
python export.py --weights weights/yolov5s.pt --img 640 --batch 1 --include onnx<br>

## 2.配置rknn-toolkit2环境<br>
主要完成linux下的onnx转rknn模型。 <br>
cd rknn-toolkit2<br>
pip install -r requirements_cp39-1.6.0.txt -i https://mirror.baidu.com/pypi/simple<br>
pip install rknn_toolkit2-1.6.0+81f21f4d-cp39-cp39-linux_x86_64.whl<br>
验证rknn-toolkit2是否安装成功：<br>
(rknn)@:./rknn-toolkit2$:python<br>
from rknn.api import RKNN<br>

没有报错说明安装rknn-toolkit2成功，并quit()退出当前编译<br>
运行以下命令将onnx转换为rknn模型<br>
python onnx2rknn.py<br>

## 3.配置rknn-toolkit-lite2环境并推理<br>
以下是基于RK3588的arm架构板卡推理流程<br>
安装rknn-toolkit-lite2<br>
进入到yolov5主目录下，运行以下命令<br>
pip install rknn_toolkit_lite2-1.4.0-cp39-cp39-linux_aarch64.whl<br>
板端推理测试，会调用librknnrt.so库，该库是一个板端的runtime库。本项目用到的库版本是1.5.2并能成功运行。rknpu2工程可以从https://github.com/rockchip-linux/rknpu2 获取。<br>
注意RKNN-Toolkit2，RKNPU2 runtime库不同版本号可能会不兼容，可能会出现 Invalid RKNN model verslon 6 错误, 请更新librknnrt.so库或者使用对应版本的RKNN-Toolkit2重新转换出rknn模型。<br>
sudo cp librknnrt.so /usr/lib/<br>

验证自带模型<br>
python test.py<br>

验证自己的模型<br>
python myrknn.py<br>

参考来源：https://www.jianshu.com/p/fd8e5da1253e<br>
参考来源：https://blog.csdn.net/qq_30841655/article/details/129836860<br>
