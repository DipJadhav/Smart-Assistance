import argparse
import warnings
import datetime
import imutils
import json
import numpy as np
import os
import time
import cv2
from threading import Thread

import tensorflow as tf
import sys
from queue import Queue
q = Queue(maxsize=0)


sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


print("[INFO] Kicking off script - " +
      datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--conf", required=True,
#                help="open configuration ")
#args = vars(ap.parse_args())

# filter warnings, load the configuration
warnings.filterwarnings("ignore")
conf = json.load(open("C:/MyTensor1/models/research/object_detection/conf.json"))

# initialize the camera and grab a reference to the raw camera capture
# if the video argument is None, then we are reading from webcam
if not conf["use_ip_cam"]:
    camera = cv2.VideoCapture(0)
    camera.set(3,1280)
    camera.set(4,720)
    time.sleep(0.25)

# otherwise, we are reading from a video input
else:
    camera = cv2.VideoCapture(conf["ip_cam_addr"])
    camera.set(3,1280)
    camera.set(4,720)

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motion_counter = 0
non_motion_timer = conf["nonMotionTimer"]
fourcc = 0x00000020  # a little hacky, but works for now
writer = None
(h, w) = (None, None)
zeros = None
output = None
made_recording = False
VideoFinished = False
fst = 1

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    (grabbed, frame) = camera.read()
    

    timestamp = datetime.datetime.now()
    motion_detected = False

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print("[INFO] Frame couldn't be grabbed. Breaking - " +
              datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=conf["resizeWidth"],height=conf["resizeHeight"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        # frame.truncate(0)
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w1, h1) = cv2.boundingRect(c)
       # cv2.rectangle(frame, (x, y), (x + w1, y + h1), (0, 255, 0), 2)
        motion_detected = True

    fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
    record_fps = 10
    ts = timestamp.strftime("%Y-%m-%d_%H_%M_%S")
    time_and_fps = ts + " - fps: " + str(fps)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Motion Detected: {}".format(motion_detected), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, time_and_fps, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    # Check if writer is None TODO: make path configurable
    if writer is None:
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        
        file_path = (conf["userDir"] + "MotionWriting/{filename}.mp4")
        file_path = file_path.format(filename=filename)

        (h2, w2) = frame.shape[:2]
        writer = cv2.VideoWriter(file_path, fourcc, record_fps, (w2, h2), True)
        zeros = np.zeros((h2, w2), dtype="uint8")

    def record_video():
        # construct the final output frame, storing the original frame
        output = np.zeros((h2, w2, 3), dtype="uint8")
        output[0:h2, 0:w2] = frame

        # write the output frame to file
        writer.write(output)
        # print("[DEBUG] Recording....")

    if motion_detected:

        # increment the motion counter
        motion_counter += 1

        # check to see if the number of frames with motion is high enough
        if motion_counter >= conf["min_motion_frames"]:
            if conf["create_image"]:
                # create image TODO: make path configurable
                image_path = ("C:/MyTensor1/models/research/object_detection/DetectedImages/{filename}.jpg").format(filename=filename)
                cv2.imwrite(image_path, frame)

            record_video()

            made_recording = True
            non_motion_timer = conf["nonMotionTimer"]

        if motion_counter >= 320 :
            writer.release()
                
            writer = None
            flname = (filename + ".mp4")
            before = os.path.join("C:/MyTensor1/models/research/object_detection/MotionWriting/",flname)
            after = os.path.join("C:/MyTensor1/models/research/object_detection/MotionDetected/",flname)
            os.rename(before,after)  
                     

            made_recording = False
            motion_counter = 0
            non_motion_timer = conf["nonMotionTimer"]
            print("Saved in 120 looooppppp")


    # If there is no motion, continue recording until timer reaches 0
    # Else clean everything up
    else:  # TODO: implement a max recording time
        # print("[DEBUG] no motion")
        if made_recording is True and non_motion_timer > 0:
            non_motion_timer -= 1
            # print("[DEBUG] first else and timer: " + str(non_motion_timer))
            record_video()
        else:
            # print("[DEBUG] hit else")
            motion_counter = 0

            if non_motion_timer == 0 :
                writer.release()
                
                writer = None
                flname = (filename + ".mp4")
                before = os.path.join("C:/MyTensor1/models/research/object_detection/MotionWriting/",flname)
                after = os.path.join("C:/MyTensor1/models/research/object_detection/MotionDetected/",flname)
                os.rename(before,after) 
                print("Saved in  Non motion timmmmmeeeeerr")
               
            
            if writer is not None:
                # print("[DEBUG] hit if 1")
                writer.release()
                
                writer = None
                
                
                #if fst == 1 :
                    #exec(compile(open("C:/MyTensor1/models/research/object_detection/Object_detection_video.py","rb").read(),"C:/MyTensor1/models/research/object_detection/Object_detection_video.py",'exec'))
                    #worker = Thread(target=ExecMe)
                    #worker.setDaemon(True)
                    #worker.start()


                           
            if made_recording is False:
                # print("[DEBUG] hit if 2")
                os.remove(file_path)
            made_recording = False
            non_motion_timer = conf["nonMotionTimer"]


            

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break


                    
# cleanup the camera and close any open windows
print("[INFO] cleaning up...")
camera.release()
cv2.destroyAllWindows()
