#! /usr/bin/python3.7
import time
import pigpio
import os #get directory
import cv2
import numpy as np #reshape the image
import picamera #capture image
from tflite_runtime.interpreter import Interpreter #tensorflow lite interpreter
from object_detection import Count

def i2c(id, tick):
    global pi
    global interpreter
    global labels

    s, b, d = pi.bsc_i2c(I2C_ADDR)
    if b and (d[-4:-1]).decode('utf-8') == "100":
        print('working')
        print(str(Count(interpreter, labels)) + '\n')

# Specify pre-trained model
MODEL_NAME = "coco_ssd_mobilenet_v1"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# First label is '???', which has to be removed
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model
interpreter = Interpreter(model_path=PATH_TO_CKPT)

# TensorFlow Lite preplans tensor allocations to optimize inference
interpreter.allocate_tensors()




SDA=18
SCL=19

I2C_ADDR=9

pi = pigpio.pi()

if not pi.connected:
    exit()

# Add pull-ups in case external pull-ups haven't been added

pi.set_pull_up_down(SDA, pigpio.PUD_UP)
pi.set_pull_up_down(SCL, pigpio.PUD_UP)

# Respond to BSC slave activity

e = pi.event_callback(pigpio.EVENT_BSC, i2c)

pi.bsc_i2c(I2C_ADDR) # Configure BSC as I2C slave

while True:
    continue

e.cancel()

pi.bsc_i2c(0) # Disable BSC peripheral

pi.stop()

