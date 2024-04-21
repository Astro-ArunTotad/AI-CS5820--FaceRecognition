#############
from __future__ import division
##############
import sys
from contextlib import contextmanager
from os import path, mkdir, listdir
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QSize, QTimer, QStringListModel, Qt, \
    QItemSelectionModel
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QHBoxLayout, \
    QShortcut, QVBoxLayout, QListView, QPushButton, QLineEdit, QGroupBox, \
    QStyledItemDelegate

import data_provider
from model import PCALDAClassifier
#######

#!/usr/bin/python
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import *



######

class NoFacesError(Exception):
    pass


class MultipleFacesError(Exception):
    pass


class CapitalizeDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        string = super().displayText(value, locale)
        return string.capitalize()


class MainApp(QWidget):
    face_detect=False
    auth=False
    total=0
    m=0
    i=0

    STRANGER_DANGER = 200
    IMAGE_SIZE = (100, 100)

    stranger_color = (179, 20, 20)
    recognized_color = (59, 235, 62)

    def __init__(self, fps=30, parent=None):
        # type: (int, Optional[QWidget]) -> None
        super().__init__(parent=parent)

        self.pkg_path = path.dirname(path.dirname(path.abspath(__file__)))
        self.training_data_dir = path.join(self.pkg_path, 'train')
        self.models_dir = path.join(self.pkg_path, 'models')
        self.model_fname = 'fisherfaces.p'

        try:
            self.model = data_provider.load_model(
                path.join(self.models_dir, self.model_fname))
        except AssertionError:
            self.model = None

        self.existing_labels = QStringListModel(self.get_existing_labels())

        self.fps = fps
        self.video_size = QSize(640, 480)

        self.gray_image = None
        self.detected_faces = []

        # Setup the UI
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)

        self.control_layout = QVBoxLayout()
        self.control_layout.setSpacing(8)
        self.main_layout.addItem(self.control_layout)

        # Setup the existing label view
        self.labels_view = QListView(parent=self)
        self.labels_view.setModel(self.existing_labels)
        self.labels_view.setSelectionMode(QListView.SingleSelection)
        self.labels_view.setItemDelegate(CapitalizeDelegate(self))
        self.control_layout.addWidget(self.labels_view)

        self.new_label_txt = QLineEdit(self)
        self.new_label_txt.returnPressed.connect(self.add_new_label)
        self.new_label_txt.returnPressed.connect(self.new_label_txt.clear)
        self.control_layout.addWidget(self.new_label_txt)

        self.add_button = QPushButton('Add New Label for New Face', self)
        self.add_button.clicked.connect(self.add_new_label)
        self.control_layout.addWidget(self.add_button)

        # Add take picture shortcut
        self.take_picture_btn = QPushButton('Take picture', self)
        self.take_picture_btn.clicked.connect(self.take_picture)
        self.control_layout.addWidget(self.take_picture_btn)
        shortcut = QShortcut(QKeySequence('Space'), self, self.take_picture)
        shortcut.setWhatsThis('Take picture and add to training data.')

        # Setup the training area
        train_box = QGroupBox('Train', self)
        train_box_layout = QVBoxLayout()
        train_box.setLayout(train_box_layout)
        self.control_layout.addWidget(train_box)
        self.train_btn = QPushButton('Train', self)
        self.train_btn.clicked.connect(self.train)
        train_box_layout.addWidget(self.train_btn)
        
        #Add Authenticate button
        auth_box = QGroupBox('Authentication', self)
        auth_box_layout = QVBoxLayout()
        auth_box.setLayout(auth_box_layout)
        self.control_layout.addWidget(auth_box)
        self.auth_btn = QPushButton('Authenticate', self)
        self.auth_btn.clicked.connect(self.demo)
        auth_box_layout.addWidget(self.auth_btn)

        self.control_layout.addStretch(0)

        # Add quit shortcut
        shortcut = QShortcut(QKeySequence('Esc'), self, self.close)
        shortcut.setWhatsThis('Quit')

        # Setup the main camera area
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.video_size)
        self.main_layout.addWidget(self.image_label)

        # Setup the camera
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(int(1000 / self.fps))

    #This method performs face classification using a loaded model.
    def classify_face(self, image): 
        if self.model is None:
            return

        label_idx, distances = self.model.predict(image.ravel(), True) #Pedicts the label of a given face image using the loaded model.

        label_idx, distance = label_idx[0], distances[0][label_idx] #If the distance from the predicted label is greater than STRANGER_DANGER, the face is labeled as a stranger.
        
        labels = self.existing_labels.stringList()
        return labels[label_idx], distance # Otherwise, it returns the predicted label and distance.

    #This method reads training images from disk and prepares them for training.
    def get_training_data(self):
        """Read the images from disk into an n*(w*h) matrix."""
        return data_provider.get_image_data_from_directory(
            self.training_data_dir)

    def train(self):
        X, y, mapping = self.get_training_data()
        # Inspect scree plot to determine appropriate number of PCA components
        classifier = PCALDAClassifier(
            n_components=2, pca_components=200, metric='euclidean',
        ).fit(X, y)

        # Replace the existing running model
        self.model = classifier

        # Save the classifier to file
        data_provider.save_model(
            classifier, path.join(self.models_dir, self.model_fname))

    def add_new_label(self):
        new_label = self.new_label_txt.text()
        new_label = new_label.lower()

        # Prevent empty entries
        if len(new_label) < 3:
            return

        string_list = self.existing_labels.stringList()

        if new_label not in string_list:
            string_list.append(new_label)
            string_list.sort()
            self.existing_labels.setStringList(string_list)

            # Automatically select the added label
            selection_model = self.labels_view.selectionModel()
            index = self.existing_labels.index(string_list.index(new_label))
            selection_model.setCurrentIndex(index, QItemSelectionModel.Select)
     
    def demo(self):
        self.auth=True
    def bblink(self,frame):

        predictor_path = 'shape_predictor_68_face_landmarks.dat_2'
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        interpolation=cv2.INTER_AREA
        
        width=120
        global ratio
        w, h = frame_grey.shape
        ratio = width / w
        height = int(h * ratio)

        frame_resized = cv2.resize(frame_grey, (height, width), interpolation)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(frame_resized, 1)
        if len(dets) > 0:
            d=dets[0]
            shape = predictor(frame_resized, d)




            coords = np.zeros((68, 2), dtype=int)
            for i in range(36,48):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            shape =coords



            
            leftEye= shape[lStart:lEnd]
            rightEye= shape[rStart:rEnd]


            A = dist.euclidean(leftEye[1], leftEye[5])
            B = dist.euclidean(leftEye[2], leftEye[4])
            
                # compute the euclidean distance between the horizontal
                # eye landmark (x, y)-coordinates
            C = dist.euclidean(leftEye[0], leftEye[3])
            #print (A,B,C)
                # compute the eye aspect ratio
            leftEAR = (A + B) / (2.0 * C)
        
            A = dist.euclidean(rightEye[1], rightEye[5])
            B = dist.euclidean(rightEye[2], rightEye[4])
            
                # compute the euclidean distance between the horizontal
                # eye landmark (x, y)-coordinates
            C = dist.euclidean(rightEye[0], rightEye[3])
            #print (A,B,C)
                # compute the eye aspect ratio
            rightEAR = (A + B) / (2.0 * C)

        
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
                
            rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear>.29:
                #print (ear)
                self.m=1
                #print ('o')
                #cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if self.m==1:
                    self.total+=1
                    #print("total",self.total)
                    self.m=0
                    #cv2.putText(frame, "blink" ,(250, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)

############################################################################################### 
    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget."""
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use the Viola-Jones face detector to detect faces to classify
        face_cascade = cv2.CascadeClassifier(path.join(
            self.pkg_path, 'resources', 'haarcascade_frontalface_default.xml'))
        self.gray_image = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in self.detected_faces:
            # Label the detected face as per the model
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, self.IMAGE_SIZE)

            result = self.classify_face(face)
            # If a model is loaded, we can predict
            if result:
                predicted, distance = self.classify_face(face)
                #print(distance)
                #print()
                if distance > self.STRANGER_DANGER:
                    predicted = 'Stranger danger!'
                    color = self.stranger_color
                    if self.auth :
                        print("Unable to identify the face, please train the model if new user for Authentication!")
                        print("------------------------------------------------------------------------------------------------")
                        self.auth=False
                else:
                    predicted = predicted.capitalize()
                    color = self.recognized_color
                    if self.auth :
                        self.i+=1
                        #thread1 = Thread(target = self.bblink, args = (frame,))
                        #thread1.start()
                        blink_cnt=self.bblink(frame)
                        if self.i==5:
                            self.i=0
                            self.auth=False
                            print("Face recognition successfull but user did not blink their eyes. Blink Detection couldn't be completed!")
                            print("------------------------------------------------------------------------------------------------")
                        if self.total>=1:
                            self.i=0
                            self.total=0
                            self.auth=False
                            print("Face recognition successfull and eye blink detected. Hence Authentication successful")
                            print("------------------------------------------------------------------------------------------------")
                            
                            
                            
                        

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = '%s (%.1f)' % (predicted, distance)
                cv2.putText(frame, text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            else:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              self.stranger_color, 2)
                cv2.putText(frame, 'Stranger danger!', (x, y + h + 15),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, self.stranger_color)

        # Display the image in the image area
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    @contextmanager
    def stop_camera_feed(self):
        """Temporarly stop the feed and face detection."""
        try:
            self.timer.stop()
            yield
        finally:
            self.timer.start(int(1000 / self.fps))

    def take_picture(self):
        # Notify the user there were no faces detected
        if self.detected_faces is None or len(self.detected_faces) < 1:
            return
            raise NoFacesError()

        if len(self.detected_faces) > 1:
            return
            raise MultipleFacesError()

        with self.stop_camera_feed():
            x, y, w, h = self.detected_faces[0]

            face = self.gray_image[y:y + h, x:x + w]
            face = cv2.resize(face, self.IMAGE_SIZE)
            denoised_image = cv2.fastNlMeansDenoising(face)

            if not self.selected_label:
                return

            self.save_image(denoised_image, self.selected_label)

    @property
    def selected_label(self):
        index = self.labels_view.selectedIndexes()
        if len(index) < 1:
            return None

        label = self.existing_labels.data(index[0], Qt.DisplayRole)

        return label

    def get_existing_labels(self):
        """Get a list of the currently existing labels"""
        return data_provider.get_folder_names(self.training_data_dir)

    def save_image(self, image: np.ndarray, label: str) -> None:
        """Save an image to disk in the appropriate directory."""
        if not path.exists(self.training_data_dir):
            mkdir(self.training_data_dir)

        label_path = path.join(self.training_data_dir, label)
        if not path.exists(label_path):
            mkdir(label_path)

        existing_files = listdir(label_path)
        existing_files = map(lambda p: path.splitext(p)[0], existing_files)
        existing_files = list(map(int, existing_files))
        last_fname = sorted(existing_files)[-1] if len(existing_files) else 0

        fname = path.join(label_path, '%03d.png' % (last_fname + 1))
        cv2.imwrite(fname, image)
           


if __name__ == "__main__":
    app = QApplication(sys.argv) #nitializes a Qt application instance app using QApplication, which is part of the PyQt framework. sys.argv is a list in Python, which contains the command-line arguments passed to the script. PyQt applications typically require this argument list for initialization.
    win = MainApp()
    win.show() #The show() method is a PyQt function that makes the window visible to the user.
    sys.exit(app.exec_()) #This line starts the event loop of the PyQt application by calling app.exec_(). The event loop continuously listens for user events (such as mouse clicks or keyboard inputs) and handles them appropriately. The exec_() method is necessary for running the event loop. The sys.exit() function ensures that the Python interpreter exits properly when the PyQt application is closed. It takes the exit status returned by app.exec_() as an argument, which typically indicates the reason for the application's termination.
