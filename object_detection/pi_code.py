#! /usr/bin/python3.7
import time
import pigpio
import os #get directory
import cv2
import numpy as np #reshape the image
import picamera #capture image
from tflite_runtime.interpreter import Interpreter #tensorflow lite interpreter
from count import Count
import dweepy

# Func: a call back function triggerred whenever i2c message coming in
def i2c(id, tick):
    global pi
    global interpreter
    global labels
    global LOG_SAVE
    global WIFI_UPLOAD
    global count
    global DEVICE_ID

    s, b, d = pi.bsc_i2c(I2C_ADDR)
    if b:
        if d.decode('utf-8') == "count\n": # arduino ask for number of people
            count = Count(interpreter, labels)
            pi.bsc_i2c(I2C_ADDR, str(count).zfill(2))
        elif d.decode('utf-8') == "down\n": # time to shutdown
            time.sleep(2)
            os.system('shutdown -h now')
        elif d.decode('utf-8') == "ack\n": # GSM successfully upload the message
            LOG_SAVE = 0
        elif d.decode('utf-8') == "nack\n": # GSM fail to upload the message, and ask for RPi to save the data temporarily
            LOG_SAVE = 1
        elif d.decode('utf-8') == "upload\n": # arduino ask RPi to upload the count+timestamp through wifi
            WIFI_UPLOAD = 1
        elif d[0:2].decode('utf-8') == "20": # get the timestamp from the arduino
            if LOG_SAVE == 1:
                with open("/home/pi/time.log", "a") as f:
                    f.write("Time: " + d.decode('utf-8') + ", Count: " + str(count))
                    f.write("\n")
                LOG_SAVE = 0
            elif WIFI_UPLOAD == 1:
                timeStamp = d.decode('utf-8')
                dweet = {'time':timeStamp, 'count':count, 'id':DEVICE_ID}
                ret = dweepy.dweet_for('gsm_mod', dweet)
                WIFI_UPLOAD = 0



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

# Flags
LOG_SAVE = 0
WIFI_UPLOAD = 0

# Global reg for latest count of people
count = 0

''' Initialize Raspberry Pi as a I2C slave '''
DEVICE_ID = 1
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

