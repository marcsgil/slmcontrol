import cv2 as cv
import screeninfo
import numpy as np
import threading
import queue


class SLMdisplay:
    def __init__(self, monitor=-1, display_interval=150, window_name='slm'):
        """Initializes an object of the class SLMdisplay. EXPERIMENTAL

        Args:
            monitor (int, optional): index that identifies the monitor in which the holograms will be shown. The primary monitor has index 0. Defaults to -1.
            display_interval (int, optional): Interval in miliseconds between each frame. Defaults to 150.
            window_name (str, optional): Name for the window. It is used as an identifier. Defaults to 'slm'.
        """
        self.monitor = screeninfo.get_monitors()[monitor]
        self.window_name = window_name
        self.display_interval = display_interval
        self.queue = queue.Queue()

        self.queue.put(
            np.zeros((self.monitor.height, self.monitor.width), dtype='uint8'))

        self.is_open = True

        thread = threading.Thread(target=self.run)
        thread.start()

    def run(self):
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.moveWindow(self.window_name, self.monitor.x - 1, self.monitor.y - 1)
        cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN,
                             cv.WINDOW_FULLSCREEN)

        while self.is_open:
            image = self.queue.get()

            cv.imshow(self.window_name, image)
            cv.waitKey(self.display_interval)

            self.queue.task_done()

            if self.queue.empty():
                self.queue.put(image)

    def updateArray(self, array):
        """Update the SLM monitor with the supplied array.

        Args:
            array (array_like): the array representing the mask that will be sent to the SLM.

        """
        self.queue.put(array)

    def close(self):
        """Closes the window associated with the SLMdisplay object.
        """
        self.is_open = False
        cv.destroyWindow(self.window_name)
