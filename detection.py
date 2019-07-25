import os
import cv2
import time
import random
import numpy as np
import pandas as pd
from cv2 import dnn
from pytesseract import image_to_string

inWidth = 720
inHeight = 1024
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
buffer = 8
classNames = ('background','plate')
net = dnn.readNetFromCaffe("MobileNetSSD_test.prototxt","lpr.caffemodel")

def detect(cpp):
    global result
    frame = cv2.imread(cpp)
    blob = dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), meanVal)
    net.setInput(blob)
    t0 = time.time()
    detections = net.forward()

    cols = frame.shape[1]
    rows = frame.shape[0]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))

    y1 = (rows - cropSize[1]) / 2
    y2 = y1 + cropSize[1]
    x1 = (cols - cropSize[0]) / 2
    x2 = x1 + cropSize[0]

    cols = frame.shape[1]
    rows = frame.shape[0]
    i = 1
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            #expand box
            xLeftBottom = xLeftBottom - buffer
            yLeftBottom = yLeftBottom - buffer
            xRightTop = xRightTop + buffer
            yRightTop = yRightTop + buffer

            if xLeftBottom < 0:
                xLeftBottom = 0
            if yLeftBottom < 0:
                yLeftBottom = 0

            crop_img = frame[yLeftBottom:yRightTop,xLeftBottom:xRightTop]
            # cv2.imshow(cpp, crop_img)
            taxi = process_img(crop_img)

            if taxi:
                text = image_to_string(crop_img)
            else:
                text = image_to_string(cv2.bitwise_not(crop_img))

            result = process_text(text)

    return result

def process_img(image):
    rgb = bincount_app(image)
    taxi = check_if_its_taxi(rgb)
    return taxi

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

def check_if_its_taxi(rgb):
    v = 0
    for p in rgb:
        v += p
    if v/3 > 150:
        return True
    else:
        return False

def process_text(text):
    L = list(text)
    for t in text:
        if t not in char:
            L.remove(t)
    return ''.join(L)

test_dir = "test_img/"
char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6',
        '7', '8', '9', '0']

for f in os.listdir(test_dir):
    number = detect(test_dir + f)
    print('True: ' + f + ' Result: ' + number)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
