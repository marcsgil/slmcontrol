import cv2 as cv
import screeninfo

class SLMdisplay:
    def __init__(self, monitor=1):
        self.monitor = screeninfo.get_monitors()[monitor]
        