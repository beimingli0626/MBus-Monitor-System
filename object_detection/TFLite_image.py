import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import matplotlib.pyplot as plt
import color

# Import packages for RTC
#import busio
#import adafruit_pcf8523
#import board

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming Sfrom the Picamera"""
    def __init__(self,resolution=(1280,720),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--image', default='frame0')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
image = args.image

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Path to image file
PATH_TO_IMAGE = '/home/pi/Projects/Python/tflite/MBus_monitor/object_detection/' + image
print(PATH_TO_IMAGE)

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1] # 300 for ssd model
width = input_details[0]['shape'][2] # 300 for ssd model

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=5).start()
time.sleep(1)

# Initialize I2C and rtc module
#rtcI2C = busio.I2C(board.SCL, board.SDA)
#rtc = adafruit_pcf8523.PCF8523(rtcI2C)
#days = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")

# Create window
#cv2.namedWindow('Crowd Counting', cv2.WINDOW_NORMAL)
#cv2.namedWindow('00', cv2.WINDOW_NORMAL)
#cv2.namedWindow('10', cv2.WINDOW_NORMAL)
#cv2.namedWindow('01', cv2.WINDOW_NORMAL)
#cv2.namedWindow('11', cv2.WINDOW_NORMAL)

j = 1
while j is 1:
    # i = 0
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = cv2.imread(PATH_TO_IMAGE)
    frame1 = color.WhiteBalance(frame1, 5)
    frame = frame1.copy()
    imH = frame.shape[0]
    imW = frame.shape[1]
    #print(frame.shape)
    #frame[:,:,3] = frame[:,:,3] - 10
    
    # crop the frame into four parts 
    frames = [frame[0:int(imH/2),0:int(imW/2)], frame[0:int(imH/2),int(imW/2):imW], frame[int(imH/2):imH,0:int(imW/2)], frame[int(imH/2):imH,int(imW/2):imW]]
    people_cnt_total=0
    for f_idx in range(4):
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frames[f_idx], cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
 
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        people_cnt = 0
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH/2)))
                xmin = int(max(1,(boxes[i][1] * imW/2)))
                ymax = int(min(imH/2,(boxes[i][2] * imH/2)))
                xmax = int(min(imW/2,(boxes[i][3] * imW/2)))

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == 'person':
                    cv2.rectangle(frames[f_idx], (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frames[f_idx], (xmin, label_ymin-labelSize[1]-10), (xmin + labelSize[0], label_ymin + baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frames[f_idx], label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    # Draw circle in center
                    xcenter = xmin + (int(round((xmax - xmin) / 2)))
                    ycenter = ymin + (int(round((ymax - ymin) / 2)))
                    cv2.circle(frames[f_idx], (xcenter, ycenter), 5, (0,0,255), thickness=-1)

                    # Print info
                    #print('People ' + str(i) + ': ' + object_name + ' at (' + str(xcenter) + ', ' + str(ycenter) + ')')
                
                    # Update the number of people
                    people_cnt_total += 1
                    people_cnt += 1

        # Show the number of people in the frame
        #print("Number of People: %d." % (people_cnt))

        # Draw framerate in corner of frame
        cv2.putText(frames[f_idx],'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        #cv2.imshow('Crowd Counting', frames[f_idx])
        
        #cv2.imshow('00', frames[0])
        #cv2.imshow('10', frames[2])
        #cv2.imshow('01', frames[1])
        #cv2.imshow('11', frames[3])

    # t = rtc.datetime
    # print("Number of People: %d. Photo taken at: %s %d/%d/%d %d:%02d:%02d" % (people_cnt_total, days[t.tm_wday], t.tm_mday, t.tm_mon, t.tm_year, t.tm_hour, t.tm_min, t.tm_sec))
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    time.sleep(5)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
#cv2.destroyAllWindows()
videostream.stop()
