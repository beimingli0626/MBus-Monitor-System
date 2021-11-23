#! /usr/bin/python3.7
import time
import pigpio
import os #get directory
import cv2
import numpy as np #reshape the image
import picamera #capture image
from tflite_runtime.interpreter import Interpreter #tensorflow lite interpreter
from count import Count
#from object_detection import Count

# Func: a call back function triggerred whenever i2c message coming in
def i2c(id, tick):
    global pi
    global interpreter
    global labels

    s, b, d = pi.bsc_i2c(I2C_ADDR)
    if b:
        if d.decode('utf-8') == "count\n":
            print('working')
            count = Count(interpreter, labels)
            print(count)
            pi.bsc_i2c(I2C_ADDR, str(count).zfill(2))
        elif d.decode('utf-8') == "down\n":
            print('shutting down')
            time.sleep(2)
            os.system('shutdown -h now')


''' Initialize TFLite Interpreter '''
MODEL_NAME = "coco_ssd_mobilenet_v1"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"

CWD_PATH = "/home/pi/Projects/MBus_monitor/object_detection" # Get path to current working directory
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME) # Path to .tflite file, which contains the model that is used for object detection
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME) # Path to label map file

with open(PATH_TO_LABELS, 'r') as f: # Load the label map
    labels = [line.strip() for line in f.readlines()]
    del(labels[0]) # remove the first label which is '???'

# Load the Tensorflow Lite model
interpreter = Interpreter(model_path=PATH_TO_CKPT)

# TensorFlow Lite preplans tensor allocations to optimize inference
interpreter.allocate_tensors()


''' Initialize Raspberry Pi as a I2C slave '''
SDA=18
SCL=19
I2C_ADDR=9

pi = pigpio.pi()
if not pi.connected:
    exit()

# Add pull-ups in case external pull-ups haven't been added
pi.set_pull_up_down(SDA, pigpio.PUD_UP)
pi.set_pull_up_down(SCL, pigpio.PUD_UP)

# Configure BSC as I2C servant
pi.bsc_i2c(I2C_ADDR)

# Respond to BSC servant activity
e = pi.event_callback(pigpio.EVENT_BSC, i2c)

'''Listening the I2C input forever'''
print("Start listening I2C")
while True:
    continue

e.cancel()
pi.bsc_i2c(0) # Disable BSC peripheral
pi.stop()

