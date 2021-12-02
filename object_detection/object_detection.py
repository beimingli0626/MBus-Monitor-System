import os #get directory
import cv2
import numpy as np #reshape the image
import time #used for sleep
import picamera #capture image
from tflite_runtime.interpreter import Interpreter #tensorflow lite interpreter

# Specify pre-trained model
MODEL_NAME = "coco_ssd_mobilenet_v1"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"

# Get path to current working directory
CWD_PATH = "/home/pi/Projects/MBus_monitor/object_detection"

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

# Para: Initialized TFLite Interpreter, all the available labels
# Func: Take image, count the number of people and return
def Count(interpreter: Interpreter, labels: list) -> int:
    # Get model details, not passed through parameter to save the stack operations
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1] # 300 for ssd model
    width = input_details[0]['shape'][2] # 300 for ssd model

    # Take a test picture and save it to a fixed space, automatically close the camera after the images are took
    PATH_TO_IMAGE = "/home/pi/Projects/MBus_monitor/iphone.jpg"
    imW, imH = 1280, 720
    #with picamera.PiCamera() as camera:
        #camera.resolution = (imW, imH)
        #camera.capture(PATH_TO_IMAGE)

    # Read the frame and crop it into four
    frame = cv2.imread(PATH_TO_IMAGE)
    frames = [frame[0:int(imH/2),0:int(imW/2)], frame[0:int(imH/2),int(imW/2):imW], frame[int(imH/2):imH,0:int(imW/2)], frame[int(imH/2):imH,int(imW/2):imW]]

    # Start crowd counting, iterate through four crops
    people_cnt_total=0
    for f_idx in range(4):
        # Acquire frame and resize to expected shape [1x300x300x3], which is specified by ssd_mobilenet
        frame_rgb = cv2.cvtColor(frames[f_idx], cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
 
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] >= 0.48) and (scores[i] <= 1.0)):
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == 'person':
                    # Get bounding box coordinates and draw box
                    ymin = int(max(1,(boxes[i][0] * imH/2)))
                    xmin = int(max(1,(boxes[i][1] * imW/2)))
                    ymax = int(min(imH/2,(boxes[i][2] * imH/2)))
                    xmax = int(min(imW/2,(boxes[i][3] * imW/2)))
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
       
                    # Update the number of people
                    people_cnt_total += 1

    # combine croped frames into single output frame
    output = frame.copy()
    output[0:int(imH/2),0:int(imW/2)] = frames[0]
    output[0:int(imH/2),int(imW/2):imW] = frames[1]
    output[int(imH/2):imH,0:int(imW/2)] = frames[2]
    output[int(imH/2):imH,int(imW/2):imW] = frames[3]
    cv2.imwrite('output.jpeg', output)
    return people_cnt_total

Count(interpreter, labels)
