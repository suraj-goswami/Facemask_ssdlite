# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck_mask
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/ssdlite_frozen_inference_graph2.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

NUM_CLASSES = 3
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a=b=0

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_box_on_image(num_face_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b,c
    face_cnt=0
    c=0
    box_count = 0
    box_list = []
    color = None
    color0 = (255,0,0)
    color1 = (0,50,255)
    color2 = (50,50,255)
    for i in range(num_face_detect):

        if (scores[i] > score_thresh):

            #print(f"frame{i}")
            #no_of_times_hands_detected+=1
            #b=b+1
            #b=1
            #print(b)
            if classes[i] == 1: 
                id = 'mask'
                #b=1
            
                
            if classes[i] == 2:
                id = 'without_mask'
                avg_width = 3.0 # To compensate bbox size change
                #b=1

            if classes[i] == 3:
                id = 'mask_weared_incorrect'
                avg_width = 3.0

            if i == 0: color = color0
            elif i==1:
                color = color1
            else:
                color = color2

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            # print("boxes[i][1] , boxes[i][3] ,"
            #       "boxes[i][0] , boxes[i][2] : ",(boxes[i][1] , boxes[i][3] ,
            #                               boxes[i][0] , boxes[i][2] ))
            #print("left, right, top, bottom",left, right, top, bottom)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            #print("p1",p1)
            #print("p2",p2)

            dist = distance_to_camera(avg_width, focalLength, int(right-left))
            
            if dist:
                face_cnt=face_cnt+1
                box_count += 1
                box_list.append(p1)
                box_list.append(p2)
            cv2.rectangle(image_np, p1, p2, color , 3, 1)


            

            cv2.putText(image_np, 'mask '+str(i)+': '+id, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
                        (int(im_width*0.65),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
           
            #a=alertcheck.drawboxtosafeline(image_np,p1,p2,Orientation)



        if face_cnt==0 :
            b=0
            #print(" no hand")
        else :
            b=1
            #print(" hand")
    #print("box_count", box_count)
    #print("box_list", box_list)
    if box_count==2:
        a,c = alertcheck_mask.drawboxtosafeline(image_np,box_list,Orientation)
            
    return b,a,c

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
