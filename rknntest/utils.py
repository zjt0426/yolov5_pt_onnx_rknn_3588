import math
import cv2
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return x


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])
    
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    
    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2]) * 2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(640 / grid_h)

    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_confidences[..., 0] >= 0.5)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]*box_confidences[..., 0][pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.6)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]
    # input_data (80, 80, 3, 85)+(40, 40, 3, 85)+(20, 20, 3, 85)
    boxes, classes, scores = [], [], []

    for input, mask in zip(input_data, masks):
        # print('dui:', input.shape)  # dui: (80, 80, 3, 85) dui: (40, 40, 3, 85) dui: (20, 20, 3, 85)
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)


    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # b = b.astype(np.float32)
        # s = s.astype(np.float32)
        # keep = nms_boxes(b, s, NMS_THRESH)

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    if not nclasses and not nscores:
        return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

def line_rate(fov, init_ang, pitch, cam_h, line_dist):
    fov_bottom = math.radians(fov / 2 + init_ang + pitch)
    line_bottom = math.atan(cam_h / line_dist)
    rate = (math.tan(fov_bottom / 2) - math.tan(fov_bottom / 2 - (fov_bottom - line_bottom))) / (
            math.tan(fov_bottom / 2) * 2)
    return rate

# def change_message(message, alarm_num, warn_num):
#     string_data = '/'.join(['0x{:02x}'.format(byte) for byte in message])
#     list_data = list(string_data)
#     alarm_num, warn_num = hex(alarm_num)[2:].zfill(2), hex(warn_num)[2:].zfill(2)
#     list_data[32:34], list_data[37:39] = str(alarm_num), str(warn_num)
#     string_data = ''.join(list_data)
#     message_toCANET = [int(value, 16) for value in string_data.split('/')]
#     message_toCANET = bytes(message_toCANET)
#     # print(message_toCANET[5])
#     # print(message_toCANET[6])
#     # print(message_toCANET[7])
#     return message_toCANET

def change_message(message, alarm_num, warn_num,this_result_warn):
    logo = 1
    string_data = '/'.join(['0x{:02x}'.format(byte) for byte in message])
    list_data = list(string_data)
    alarm_num, warn_num, logo = hex(alarm_num)[2:].zfill(2), hex(warn_num)[2:].zfill(2), hex(logo)[2:].zfill(2)
    list_data[32:34], list_data[37:39] = str(alarm_num), str(warn_num)
    if this_result_warn == 1:
        list_data[27:29] = str(logo)
    string_data = ''.join(list_data)
    message_toCANET = [int(value, 16) for value in string_data.split('/')]
    message_toCANET = bytes(message_toCANET)
    # print(message_toCANET[5])
    # print(message_toCANET[6])
    # print(message_toCANET[7])
    return message_toCANET

def draw(image, boxes, scores, classes, setLeftTop_alarm, setRightTop_alarm,
         setLeftBottom_alarm, setRightBottom_alarm,
         setLeftTop_warn, setRightTop_warn, setLeftBottom_warn, setRightBottom_warn,
         camDist, camHigh, fov, init_ang, camPitch, camRoll, camCourse, warn_dist, alarm_dist):
    num_sale = 0
    num_warn = 0
    num_alarm = 0
    for box, score, cl in zip(boxes, scores, classes):
        '''识别框的得分小于阈值的，过滤掉，有效防止低得分的干扰'''
        if score <= 0.8:
            continue
        left, top, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(left, top, right, bottom))
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        '''识别框|得分绘制'''
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format("person", score),
                    (left, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # two region points change -------------
        # setLeftTop_alarm,setRightTop_alarm,setLeftTop_warn,setRightTop_warn,angle0,alarm_dist,warn_dist
        '''根据视差角|相机初始角|掘进机俯仰角|相机高度|报警距离 计算报警线在画面的位置比例'''
        alarmRate = line_rate(fov, init_ang, camPitch, camHigh, alarm_dist)
        # print('alarmrate:', alarmRate)
        '''针对384*288 letterbox后resize到640*640的情况'''
        setLeftTop_alarm, setRightTop_alarm = list(setLeftTop_alarm), list(setRightTop_alarm)
        setLeftTop_alarm[1], setRightTop_alarm[1] = int(640 - (480 * alarmRate + 80)), int(640 - (480 * alarmRate + 80))
        setLeftTop_alarm, setRightTop_alarm = tuple(setLeftTop_alarm), tuple(setRightTop_alarm)
        '''报警区域绘制颜色'''
        color = (0, 0, 255)
        alarm_region = np.array([setLeftTop_alarm, setRightTop_alarm, setRightBottom_alarm, setLeftBottom_alarm])
        cv2.polylines(image, [alarm_region], True, color, 1)
        '''xy坐标系下，框的右下坐标'''
        pic_rightbottom = (right, image.shape[0] - bottom)
        '''区域上线的方程'''
        slope_a_1 = ((image.shape[0] - setRightTop_alarm[1]) - (image.shape[0] - setLeftTop_alarm[1])) / (setRightTop_alarm[0] - setLeftTop_alarm[0])  # 方程1
        bias_b_1 = (image.shape[0] - setLeftTop_alarm[1]) - slope_a_1*setLeftTop_alarm[0]
        '''区域右线的方程'''
        slope_a_2 = ((image.shape[0] - setLeftBottom_alarm[1]) - (image.shape[0] - setLeftTop_alarm[1])) / (setLeftBottom_alarm[0] - setLeftTop_alarm[0])  # 方程2
        bias_b_2 = (image.shape[0] - setLeftTop_alarm[1]) - slope_a_2 * setLeftTop_alarm[0]
        result1 = slope_a_1 * pic_rightbottom[0] + bias_b_1
        result2 = slope_a_2 * pic_rightbottom[0] + bias_b_2
        '''根据视差角|相机初始角|掘进机俯仰角|相机高度|报警距离 计算预警线在画面的位置比例'''
        warnRate = line_rate(fov, init_ang, camPitch, camHigh, warn_dist)
        # print('warnRate:', warnRate)
        '''针对384*288 letterbox后resize到640*640的情况'''
        setLeftTop_warn, setRightTop_warn = list(setLeftTop_warn), list(setRightTop_warn)
        setLeftTop_warn[1], setRightTop_warn[1] = int(640 - (480 * warnRate + 80)), int(640 - (480 * warnRate + 80))
        setLeftTop_warn, setRightTop_warn = tuple(setLeftTop_warn), tuple(setRightTop_warn)
        '''预警区域绘制颜色'''
        color_warn = (0, 255, 255)
        warn_region = np.array([setLeftTop_warn, setRightTop_warn, setRightBottom_warn, setLeftBottom_warn])
        cv2.polylines(image, [warn_region], True, color_warn, 1)
        '''区域上线的方程'''
        slope_a_3 = ((image.shape[0] - setRightTop_warn[1]) - (image.shape[0] - setLeftTop_warn[1])) / (
                    setRightTop_warn[0] - setLeftTop_warn[0])  # 方程1
        bias_b_3 = (image.shape[0] - setLeftTop_warn[1]) - slope_a_3 * setLeftTop_warn[0]
        '''区域右线的方程'''
        slope_a_4 = ((image.shape[0] - setLeftBottom_warn[1]) - (image.shape[0] - setLeftTop_warn[1])) / (
                    setLeftBottom_warn[0] - setLeftTop_warn[0])  # 方程2
        bias_b_4 = (image.shape[0] - setLeftTop_warn[1]) - slope_a_4 * setLeftTop_warn[0]
        result3 = slope_a_3 * pic_rightbottom[0] + bias_b_3
        result4 = slope_a_4 * pic_rightbottom[0] + bias_b_4

        # print('666:', result1, result2, pic_leftbottom[1])
        '''计数规则'''
        if pic_rightbottom[1] < result1 and pic_rightbottom[1] < result2:
            # print('dangerous!')
            num_alarm += 1
        elif pic_rightbottom[1] > result3 or pic_rightbottom[1] > result4:
            # print('safe!')
            num_sale += 1
        else:
            # print('warn!')
            num_warn += 1
    return num_sale, num_warn, num_alarm,image




    #     # 20230519
    #     iou = rectangleAlarmRegion(setLeftTop, setLeftBottom, setRightTop, setRightBottom, top, left, right, bottom)
    #     # iou = trapezoidAlarmRegion(setLeftTop, setLeftBottom, setRightTop, setRightBottom, top, left, right, bottom)
    #     iou_list.append(iou)
    #
    #     # cv2.rectangle(image, setLeftTop, setRightBottom, (255, 255, 0), 2) # -----------------------------------show region rectangle
    #     cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
    #     cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
    #                 (top, left - 6),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.6, (0, 0, 255), 2)
    #     # 20230519
    #     cv2.putText(image, '{0} {1:.2f}'.format('iou', iou),
    #                 (top, left - 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.6, (0, 0, 255), 2)
    # return iou_list


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 20230519
def rectangleAlarmRegion(setLeftTop, setLeftBottom, setRightTop, setRightBottom, detectLeft, detectTop, detectRight,
                         detectBottom):
    setLeft = setLeftTop[0]
    setTop = setLeftTop[1]
    setRight = setRightBottom[0]
    setBottom = setRightBottom[1]
    width0 = setRight - setLeft
    height0 = setBottom - setTop
    width1 = detectRight - detectLeft
    height1 = detectBottom - detectTop
    max_x = max(setRight, detectRight)
    min_x = min(setLeft, detectLeft)
    width = width0 + width1 - (max_x - min_x)
    max_y = max(setBottom, detectBottom)
    min_y = min(setTop, detectTop)
    height = height0 + height1 - (max_y - min_y)
    interArea = width * height
    boxArea = width0 * height0
    detectBoxArea = width1 * height1
    iou = interArea / (boxArea + detectBoxArea - interArea)
    return iou



# 20230519
def trapezoidAlarmRegion(setLeftTop, setRightTop, setLeftBottom, setRightBottom, detectLeft, detectTop, detectRight,
                         detectBottom):
    setLeftTop, setRightBottom = (setLeftTop[0], setLeftTop[1]), (setRightTop[0], setRightBottom[1])
    setLeft = setLeftTop[0]
    setTop = setLeftTop[1]
    setRight = setRightBottom[0]
    setBottom = setRightBottom[1]
    width0 = setRight - setLeft
    height0 = setBottom - setTop
    width1 = detectRight - detectLeft
    height1 = detectBottom - detectTop
    max_x = max(setRight, detectRight)
    min_x = min(setLeft, detectLeft)
    width = width0 + width1 - (max_x - min_x)
    max_y = max(setBottom, detectBottom)
    min_y = min(setTop, detectTop)
    height = height0 + height1 - (max_y - min_y)
    interArea = width * height
    boxArea = width0 * height0
    detectBoxArea = width1 * height1
    iou = interArea / (boxArea + detectBoxArea - interArea)
    return iou

def make_streamer(pipe, rtscap):
    while (1):
        frame_ok, cur_frame = rtscap.read_frame()
        if not frame_ok:
            continue
        img = cur_frame.copy()
        '''推流'''
        pipe.stdin.write(img.tostring())