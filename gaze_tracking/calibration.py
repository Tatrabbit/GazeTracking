from __future__ import division
import cv2
from .pupil import Pupil


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds = []
        self._average_iris_size = 0.48

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self):
        """Returns the threshold value for this eye
        """
        return int(sum(self.thresholds) / len(self.thresholds))

    def get_average_iris_size(self):
        return self._average_iris_size

    def set_average_iris_size(self, value):
        if self._average_iris_size == value:
            return

        self._average_iris_size = value
        self.thresholds.clear()

    average_iris_size = property(get_average_iris_size, set_average_iris_size)

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        if nb_pixels == 0:
            return 0
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    def find_best_threshold(self, eye_frame):
        """Calculates the optimal threshold to binarize the
        frame for the given eye.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - self.average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
        """
        threshold = self.find_best_threshold(eye_frame)

        self.thresholds.append(threshold)
        if len(self.thresholds) > self.nb_frames:
            self.thresholds.pop(0)
