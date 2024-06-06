# -*- coding: UTF-8 -*-
import socket
import configparser
import requests
import time
import cv2 as cv
import os

import subprocess
import threading
import time
import cv2
import sys

import requests

from utils import letterbox, yolov5_post_process, draw
from rknnlite.api import RKNNLite
import os
import numpy as np

#some marked var
var_npuinfer = 0
var_cam = 0
var_nputrans = 0
var_cam12 = 0
var_v157 = 0
var_led = 0

from rknnlite.api import RKNNLite
rknn = RKNNLite()
var_npudriver = 1
if var_npudriver == 1:
    rknn = RKNNLite()
    #ret = rknn.load_rknn("yolov5s_for_rk3588.rknn")
    ret = rknn.load_rknn('yolov5s-0327.rknn')
    if ret != 0:
        print('Load rknn_model failed!')
    print('Load rknn_model done!')
    
    #ret = rknn.init_runtime()
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
    else:
        print('Init runtime environment done!')
        
    username = 'hkdq'
    password = 'admin@134'
    ip_address = '192.168.200.139'
    stream_url = f'rtsp://{username}:{password}@{ip_address}/stream1'
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print('cannot open camera')
        exit()
    while True:
        t0_infer = time.time()
        ret, frame = cap.read()
        if not ret:
            print('can not receive frame. Exiting ...')
            break
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (640, 640))
        
        
    
    
        #img = cv2.imread("a.jpg")
        #img, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (640, 640))
        
    
        
        outputs = rknn.inference(inputs=[input_img], data_type='uint8')
        print(outputs[0].shape)
    
        # print("---infer take time:", time.time() - t0_infer)

        input0_data = outputs[0]  # (1, 3, 80, 80, 85)
        input1_data = outputs[1]
        input2_data = outputs[2]
        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))  # (3, 80, 80, 85)
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))
        
        input_data = list()
        input_data.append(np.transpose(input0_data, (1, 2, 0, 3)))  # (80, 80, 3, 85)
        input_data.append(np.transpose(input1_data, (1, 2, 0, 3)))
        input_data.append(np.transpose(input2_data, (1, 2, 0, 3)))
        
        #input_data = list()
        #input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        #input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        #input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        print(input_data[0].shape)
        print('------------')
        # print(time.time()-t0_infer)
        
        # t0_infer = time.time()
        
        boxes, classes, scores = yolov5_post_process(input_data)
        
        

        print(classes)
        if classes is not None:

            if 0 in classes:

                img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        
                num_safe, num_warn, num_alarm,image_saved = draw(img_bgr, boxes, scores, classes,
                                                     setLeftTop_alarm=(40,400),
                                                     setRightTop_alarm=(600,400),
                                                     setLeftBottom_alarm=(20,560),
                                                     setRightBottom_alarm=(620,560),
                                                     setLeftTop_warn=(40,200),
                                                     setRightTop_warn=(600,200),
                                                     setLeftBottom_warn=(15,560),
                                                     setRightBottom_warn=(625,560),
                                                     camDist=3500, camHigh=1810, fov=28.2,
                                                     init_ang=6.5,
                                                     camPitch=0, camRoll=0,
                                                     camCourse=0,
                                                     warn_dist=17000, alarm_dist=10000)
                cv2.imwrite('aabb.jpg', image_saved)
                if num_alarm>0:
                    print("npu infer success")
                    var_npuinfer = 1
        print(time.time()-t0_infer)

cfgpath = 'config_test.ini'
algConfig = configparser.ConfigParser()
algConfig.read(cfgpath, encoding='utf-8')

board_ip = algConfig.get("info", "board_ip")
cam_url = algConfig.get("info", "cam_url")
can_host = algConfig.get("info", "can_host")
can_port = int(eval(algConfig.get("info", "can_port")))
frame_ret = None
camserver_ret = None
can_ret = None


