import os
import cv2
import time
import random
import numpy as np
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
    global result, taxi
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
            taxi, wimg = bincount_app(crop_img)
            # cv2.imshow(cpp, wimg)

            if taxi:
                text = image_to_string(wimg, config=config)
            else:
                text = image_to_string(cv2.bitwise_not(wimg), config=config)

            result = process_text(text)

            if len(result) <= 3:
                result = 'No License Plate Found'

    return result, taxi

def bincount_app(img):
    histr = np.bincount(img.ravel(),minlength=256)
    taxi = check_if_its_taxi(histr)
    wider_img = add_border(img)
    return taxi, wider_img

def check_if_its_taxi(histr):
    v1 = 0
    v2 = 0
    for i in range(0, 101):
        v1 += histr[i]
    for j in range(100, 256):
        v2 += histr[j]
    if v2 > v1:
        return True
    else:
        return False

def add_border(img):
    top = int(0.05 * img.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * img.shape[1])  # shape[1] = cols
    right = left
    dst = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE, None)
    return dst

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
config = ("-l eng --oem 1 --psm 7")

for f in os.listdir(test_dir):
    number, taxi = detect(test_dir + f)
    print('Filename: ' + f + ' Result: ' + number + ' Taxi: ' + str(taxi))

# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()
