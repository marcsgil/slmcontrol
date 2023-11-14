import cv2 as cv
import screeninfo
import numpy as np
from slmcontrol.hologram import build_grid


class SLMdisplay:
    def __init__(self, monitor=1, window_name='slm'):
        """Initializes an object of the class SLMdisplay. EXPERIMENTAL

        Args:
            monitor (int, optional): index that identifies the monitor in which the holograms will be shown. The primary monitor has index 0. Defaults to 1.
            window_name (str, optional): Name for the window. It is used as an identifier. Defaults to 'slm'.
        """
        self.monitor = screeninfo.get_monitors()[monitor]
        self.window_name = window_name

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.moveWindow(self.window_name, self.monitor.x - 1, self.monitor.y - 1)
        cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN,
                             cv.WINDOW_FULLSCREEN)
        image = np.zeros(
            (self.monitor.height, self.monitor.width), dtype='uint8')
        cv.imshow(self.window_name, image)
        cv.waitKey(0)
        # cv.destroyAllWindows()

    def updateArray(self, array, sleep=150):
        """Update the SLM monitor with the supplied array.
        Note that the array is not the same size as the SLM resolution,
        the image will be deformed to fit the screen.

        Args:
            array (array_like): the array representing the mask that will be sent to the SLM.
            sleep (Real, optional): Time in miliseconds that will be waited after calling this function.
                This is important when one shows a series of masks in sequence, in which case one must wait for the SLM to properly dislplay each mask.
                Defaults to 150.
        """
        cv.imshow(self.window_name, array)
        cv.waitKey(sleep)

    def close(self):
        """Closes the window associated with the SLMdisplay object.
        """
        cv.destroyWindows(self.window_name)
