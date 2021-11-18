from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .head_angle import HeadAngle
from .calibration import Calibration

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.gray_frame = None
        self.eye_left = None
        self.eye_right = None
        self.head = HeadAngle()

        self._calibration_left = Calibration()
        self._calibration_right = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def get_average_iris_size(self):
        return self._calibration_left.get_average_iris_size()

    def set_average_iris_size(self, value):
        self._calibration_left.set_average_iris_size(value)
        self._calibration_right.set_average_iris_size(value)

    average_iris_size = property(get_average_iris_size, set_average_iris_size)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _predict_landmarks(self):
        """Detects the face, and analyzes it for landmarks
        """
        faces = self._face_detector(self.gray_frame)
        if len(faces) == 0:
            return

        return self._predictor(self.gray_frame, faces[0])

    def refresh(self, frame):
        """Refreshes the frame, detects the face, and analyzes it.
        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        self.landmarks = self._predict_landmarks()

        self.head.refresh(self.landmarks)

        # TODO: Use the tilt of the head to rotate the frame
        if self.landmarks is not None:
            self.eye_left = Eye(self.gray_frame, self.landmarks, 0, self._calibration_left)
            self.eye_right = Eye(self.gray_frame, self.landmarks, 1, self._calibration_right)
        else:
            self.eye_left = None
            self.eye_right = None

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self, offset=0.0):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 + offset)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 + offset)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self, offset=0.0):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 + offset)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 + offset)
            return (pupil_left + pupil_right) / 2

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self, use_gray=False):
        """Returns the main frame with pupils highlighted"""
        frame = (self.gray_frame if use_gray else self.frame).copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
