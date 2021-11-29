import numpy as np
import cv2

class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    _erode_kernel = np.ones((3, 0), np.uint8)

    def __init__(self, eye_frame):
        self.frame = Pupil._process_image(eye_frame)

        x = Pupil._min_index(0, self.frame)
        y = Pupil._min_index(1, self.frame)
        self.center = (x, y)

    @property
    def horizontal_ratio(self):
        width = float(self.frame.shape[1])
        offset = float(self.center[0]) / width
        return offset - 0.5

    @property
    def vertical_ratio(self):
        height = float(self.frame.shape[0])
        offset = float(self.center[1]) / height
        return offset - 0.5

    @staticmethod
    def _process_image(eye_frame):
        """Performs operations on the eye frame to detect the pupil

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else

        Returns:
            An (x, y) tuple of the position of the "darkest" position in the input frame
        """

        # TODO play with filters to improve outcome
        new_frame = eye_frame.copy()
        #new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        #new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        #new_frame = cv2.threshold(new_frame, 80, 255, cv2.THRESH_TRUNC)[1]
        #new_frame = cv2.erode(new_frame, Pupil._erode_kernel, iterations=3)

        return new_frame

    @staticmethod
    def _min_index(axis, frame):
        sum = np.sum(frame, axis)
        min = np.min(sum)
        return np.where(sum == min)[0][0]