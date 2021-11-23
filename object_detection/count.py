import os #get directory
import cv2
import numpy as np #reshape the image
import time #used for sleep
import picamera #capture image
import io
from tflite_runtime.interpreter import Interpreter #tensorflow lite interpreter

# Para: Initialized TFLite Interpreter, all the available labels
# Func: Take image, count the number of people and return
def Count(interpreter: Interpreter, labels: list) -> int:
    # Get model details, not passed through parameter to save the stack operations
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = 300
    width = 300

    # Take a 1280*720 image as an in-memory stream, instead of saving the image
    imW, imH = 1280, 720
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.resolution = (imW, imH)
        camera.capture(stream, 'jpeg')
        
    # Restore the frame from stream and crop it into four
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, 1)
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
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
 
        # Loop over all detections
        for i in range(len(scores)):
            if ((scores[i] >= 0.48) and (scores[i] <= 1.0)):
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == 'person':
                    people_cnt_total += 1
    return people_cnt_total