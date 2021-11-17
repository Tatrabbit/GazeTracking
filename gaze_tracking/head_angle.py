import math
from .utils import *

class HeadAngle(object):
    """
    This class detects the tilt of the user's head.
    """

    TOP_POINT = 27 # Top of nose
    BOTTOM_POINT = 8 # Chin

    CHEEK_LEFT = 2
    CHEEK_RIGHT = 14

    CHEEK_POINTS_LEFT = range(0, 5, 1)
    CHEEK_POINTS_RIGHT = range(16, 11, -1)

    # Stick it on your chin or nose! 12 in each packet:
    #   https://www.youtube.com/watch?v=IPaNTxEhfzs
    #   1969 Ellisdons Novelty Catalogue | Ashens
    CHIN_POINTS = range(7, 10)
    NOSE_POINTS = range(27, 31)
    NOSTRIL_POINTS = range(31,36)

    look_ratio = 3.0

    # An estimation of how far the head can rotate before the camera
    # loses tracking. Proper trig could solve this accurately,
    # however the depth of the face would need to be known,
    # rendering such a calculation moot.
    TWIST_MAX = 80.0

    def __init__(self):
        self._tilt = None
        self._pitch = None
        self._twist = None
        self.draw_points = []
        self._known = False

    @property
    def tilt(self):
        return self._tilt

    @property
    def pitch(self):
        return self._pitch

    @property
    def twist(self):
        return self._twist

    @property
    def known(self):
        return self._known

    def refresh(self, landmarks):
        if landmarks == None:
            self._known = False
            return

        self._refresh_tilt(landmarks)
        self._refresh_twist(landmarks)
        self._refresh_pitch(landmarks)
        self._known = True

    def _refresh_tilt(self, landmarks):
        top = landmarks.part(self.TOP_POINT)
        bottom = landmarks.part(self.BOTTOM_POINT)

        difference_x = top.x - bottom.x
        difference_y = top.y - bottom.y

        radians = math.atan2(difference_y, difference_x)
        radians = max(-math.pi, min(radians, math.pi)) # Clamp
        self._tilt = math.degrees(radians) + 90.0

    def _refresh_pitch(self, landmarks):
        nostrils = self._get_average(landmarks, self.NOSTRIL_POINTS)
        def get_position(point):
            part = landmarks.part(point)
            return (part.x, part.y)

        leftmost = get_position(self.CHEEK_LEFT)
        rightmost = get_position(self.CHEEK_RIGHT)

        cheeks = (leftmost[1] + rightmost[1]) / 2.0
        face_size = abs(rightmost[0] - leftmost[0])

        self._pitch = (nostrils[1] - cheeks) / float(face_size) * self.look_ratio * 90.0

    def _refresh_twist(self, landmarks):
        left = self._get_average(landmarks, self.CHEEK_POINTS_LEFT)
        right = self._get_average(landmarks, self.CHEEK_POINTS_RIGHT)
        chin = self._get_average(landmarks, self.CHIN_POINTS)
        nose = self._get_average(landmarks, self.NOSE_POINTS)

        self.draw_points = [left, right, chin, nose]

        x = float(chin[0] + nose[0]) / 2.0
        x = inverse_lerp(right[0], left[0], x)
        self._twist = ((0.5 - x) * 2.0) * 90.0

    def _get_average(self, landmarks, points):
        x = int(sum([landmarks.part(i).x for i in points]) / len(points))
        y = int(sum([landmarks.part(i).y for i in points]) / len(points))
        return (x, y)