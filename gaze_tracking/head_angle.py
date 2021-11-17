import math

class HeadAngle(object):
    """
    This class detects the tilt of the user's head.
    """

    TOP_POINT = 27 # Top of nose
    BOTTOM_POINT = 8 # Chin

    def __init__(self):
        self._tilt = None

    @property
    def tilt(self):
        return self._tilt

    def refresh(self, landmarks):
        if landmarks == None:
            self._tilt = None
            return
        
        top = landmarks.part(self.TOP_POINT)
        bottom = landmarks.part(self.BOTTOM_POINT)

        difference_x = top.x - bottom.x
        difference_y = top.y - bottom.y

        radians = math.atan2(difference_y, difference_x)
        radians = max(-math.pi, min(radians, math.pi)) # Clamp
        self._tilt = math.degrees(radians) + 90.0