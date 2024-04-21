#!/usr/bin/python
from __future__ import division
import dlib
from imutils import face_utils
import cv2
import numpy as np
from scipy.spatial import distance as dist

# This resize function is a utility function designed to resize images while maintaining their aspect ratio.
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

###### This shape_to_np function converts the facial landmark coordinates obtained from the dlib library's facial landmark predictor into a NumPy array of coordinates.
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

###### This eye_aspect_ratio function computes the Eye Aspect Ratio (EAR) for a given set of eye landmarks. The EAR is a measure used in facial recognition and eye tracking to quantify the level of eye openness. 
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # print (A,B,C)
	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
    return ear



###Execution of code starts here ###
camera = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0
m=0
while True:
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

# Ask the detector to find the bounding boxes of each face. The 1 in the second argument indicates that we should upsample the image 1 time. This will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    # If faces are detected in the frame, this block iterates over each detected face. For each face, it uses the facial landmark predictor to detect facial landmarks (shape). Then, it extracts the left and right eye landmarks, calculates the eye aspect ratio (ear), and draws contours around the eyes on the original frame.
    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye= shape[lStart:lEnd]
            rightEye= shape[rStart:rEnd]
            leftEAR= eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
	       
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear>.29:
                m=1
                # Draw a filled rectangle as background for the text
                text = "EYES ARE OPEN"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_width, text_height = text_size
                # Coordinates for the text and background rectangle
                x, y = 10, 50
                padding = 5
                rectangle_coords = ((x - padding, y - text_height - padding), (x + text_width + padding, y + padding))
                text_coords = (x, y)
                # Draw the rectangle
                cv2.rectangle(frame, rectangle_coords[0], rectangle_coords[1], (0, 225, 100), cv2.FILLED)
                # Add text on top of the rectangle
                cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            else:
                if m==1:
                    total+=1
                    m=0
                    text = "EYE BLINK DETECTED"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_width, text_height = text_size
                    # Coordinates for the text and background rectangle
                    x, y = 10, 100
                    padding = 5
                    rectangle_coords = ((x - padding, y - text_height - padding), (x + text_width + padding, y + padding))
                    text_coords = (x, y)
                    # Draw the rectangle
                    cv2.rectangle(frame, rectangle_coords[0], rectangle_coords[1], (0, 0, 0), cv2.FILLED)
                    # Add text on top of the rectangle
                    cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw a filled rectangle as background for the text
                text = "EYES ARE CLOSED"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_width, text_height = text_size
                # Coordinates for the text and background rectangle
                x, y = 10, 50
                padding = 5
                rectangle_coords = ((x - padding, y - text_height - padding), (x + text_width + padding, y + padding))
                text_coords = (x, y)
                # Draw the rectangle
                cv2.rectangle(frame, rectangle_coords[0], rectangle_coords[1], (0, 0, 255), cv2.FILLED)
                # Add text on top of the rectangle
                cv2.putText(frame, text, text_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.putText(frame, "TOTAL EYE BLINK COUNT: {}".format(total), (250, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0, 0, 0), 3)


            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
