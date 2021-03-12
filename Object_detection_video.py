######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob
import time
from queue import Queue
q = Queue(maxsize=0)
import json

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites 
from utils import label_map_util
from utils import visualization_utils as vis_util

conf = json.load(open("C:/MyTensor1/models/research/object_detection/conf.json"))

#for entrans exit
height = 0
width = 0
OffsetRefLines = 216

# Name of the directory containing the object detection module we're using
def ExecMe():
    print("In Video Code")
    MODEL_NAME = 'inference_graph'
    #VIDEO_NAME = 'test.mov'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    

    # Number of classes the object detector can identify
    NUM_CLASSES = 4

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    list = os.listdir("C:/MyTensor1/models/research/object_detection/MotionDetected") # dir is your directory path
    number_files = len(list)
    print (number_files)
    act_num_file = number_files

    arr = os.listdir("C:/MyTensor1/models/research/object_detection/MotionDetected")
    #files=glob.glob("C:/MyTensor1/models/research/object_detection/MotionDetected")
    for file in arr:
        q.put(file)
#        print(q.get())
    #print("end q")
    
    while q.empty():
            arr = os.listdir("C:/MyTensor1/models/research/object_detection/MotionDetected")
            for file in arr:
                q.put(file)
            if q.empty():
                time.sleep(5)

    while not q.empty():

        path = q.get()
        print(path)
        # Path to video
        PATH_TO_VIDEO = os.path.join("C:/MyTensor1/models/research/object_detection/MotionDetected/",path)
        #print(PATH_TO_VIDEO)
        # Open video file
        video = cv2.VideoCapture(PATH_TO_VIDEO)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(length)

        while(video.isOpened()):

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = video.read()
            frame_expanded = np.expand_dims(frame, axis=0)

            height = np.size(frame,0)
            width = np.size(frame,1)


            CoorYEntranceLine = conf["EntranceLine"]

            CoorYExitLine = conf["ExitLine"]

            cv2.line(frame, (0,(CoorYEntranceLine)), (width,(CoorYEntranceLine)), (255, 0, 0), 2)

            cv2.line(frame, (0,(CoorYExitLine)), (width,(CoorYExitLine)), (0, 0, 255), 2)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.80)

            # All the results have been drawn on the frame, so it's time to display it.
         
            if conf["show_video_detection"]:
                cv2.imshow('Object detector', frame)
                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break
            
            length = length - 1
            if length == 1:
                break
            

            

            # Clean up
        video.release()
        cv2.destroyAllWindows()
        before = os.path.join("C:/MyTensor1/models/research/object_detection/MotionDetected/",path)
        after = os.path.join("C:/MyTensor1/models/research/object_detection/MotionDetectedTested/",path)
        os.rename(before,after )
        q.task_done()
        while q.empty():
            arr = os.listdir("C:/MyTensor1/models/research/object_detection/MotionDetected")
            for file in arr:
                q.put(file)
            if q.empty():
                time.sleep(5)
        #return True
    
        
             
                
 
ExecMe()
