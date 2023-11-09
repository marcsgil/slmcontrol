"""
Based on slmpy from Sebastien M. Popoff
https://github.com/wavefrontshaping/slmPy
"""

import wx
import threading
import numpy as np
import time

EVT_NEW_IMAGE = wx.PyEventBinder(wx.NewEventType(), 0)

class ImageEvent(wx.PyCommandEvent):
    def __init__(self, img, eventLock, eventType=EVT_NEW_IMAGE.evtType[0], id=0):
        wx.PyCommandEvent.__init__(self, eventType, id)
        self.img = img
        self.eventLock = eventLock
        eventLock.acquire()
        self.color = False       

class SLMframe(wx.Frame):
    def __init__(self, monitor):
        
        style = wx.DEFAULT_FRAME_STYLE
        self.SetMonitor(monitor)
        super().__init__(None,
                         -1,
                         'SLM window',
                         pos = (self._x0, self._y0), 
                         size = (self._resX, self._resY),
                         style = style
                        ) 
        
        self.Window = SLMwindow(self,
                                res = (self._resX, self._resY)
                               )
        self.Show()
        
        self.Bind(EVT_NEW_IMAGE, self.OnNewImage)
        self.ShowFullScreen(not self.IsFullScreen(), wx.FULLSCREEN_ALL)
        self.SetFocus()
        
    def SetMonitor(self, monitor: int):
        if (monitor < 0 or monitor > wx.Display.GetCount()-1):
            raise ValueError('Invalid monitor (monitor %d).' % monitor)
        self._x0, self._y0, self._resX, self._resY = wx.Display(monitor).GetGeometry()
        
    def OnNewImage(self, event):
        self.Window.UpdateImage(event)
        
    def Quit(self):
        wx.CallAfter(self.Destroy)
        
        
class SLMwindow(wx.Window):
    def __init__(self,  *args, **kwargs):
        self.res = kwargs.pop('res')
        kwargs['style'] = kwargs.setdefault('style', wx.NO_FULL_REPAINT_ON_RESIZE) | wx.NO_FULL_REPAINT_ON_RESIZE
        super().__init__(*args, **kwargs)
        
        # hide cursor
        cursor = wx.Cursor(wx.CURSOR_BLANK)
        self.SetCursor(cursor) 
        
        self.img = wx.Image(*self.res)
        self._Buffer = wx.Bitmap(*self.res)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(EVT_NEW_IMAGE, self.UpdateImage)
        self.Bind(wx.EVT_PAINT,self.OnPaint)
        
        self.OnSize(None)
        
    def OnPaint(self, event):
        self._Buffer = self.img.ConvertToBitmap()
        wx.BufferedPaintDC(self, self._Buffer)
 
    def OnSize(self, event):
        # Make new offscreen bitmap: this bitmap will always have the
        # current drawing in it, so it can be used to save the image to
        # a file, or whatever.
        self._Buffer = wx.Bitmap(*self.res)
        
    def UpdateImage(self, event):
        self.img = event.img
        self.Refresh(eraseBackground=False)
        event.eventLock.release()


class SLMdisplay:
    """Interface for sending images to the display frame."""
    def __init__(self,
                 monitor = 1):         
        self.monitor = monitor
        # Create the thread in which the window app will run
        # It needs its thread to continuously refresh the window
        self.vt = videoThread(self)      
        self.eventLock = threading.Lock()
        self.last_update = time.time()
        
    def getSize(self):
        return self.vt.frame._resX, self.vt.frame._resY

    def updateArray(self, array, sleep = 0.15):
        """
        Update the SLM monitor with the supplied array.
        Note that the array is not the same size as the SLM resolution,
        the image will be deformed to fit the screen.
        
        Parameters
        ----------
        array : array_like
            Numpy array to display, should be the same size as the resolution of the SLM.
        sleep : float
            Pause in seconds after displaying an image.
        """
        # create a wx.Image from the array
        h,w = array.shape
        color_array = np.stack((array,array,array), axis=2)
        data = color_array.tostring()
        img = wx.ImageFromBuffer(width=w, height=h, dataBuffer=data)

        # Create the event
        event = ImageEvent(img, self.eventLock)
        self.vt.frame.AddPendingEvent(event)
        time.sleep(sleep)
        
    def close(self):
         self.vt.frame.Quit()

class videoThread(threading.Thread):
    """Run the MainLoop as a thread. 
    WxPython is not designed for that, it will give a warning on exit, but it will work, 
    see: https://wiki.wxpython.org/MainLoopAsThread
    Access the frame with self.frame."""
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        # Set as deamon so that it does not prevent the main program from exiting
        self.setDaemon(1)
        self.start_orig = self.start
        self.start = self.start_local
        self.frame = None #to be defined in self.run
        self.lock = threading.Lock()
        self.lock.acquire() #lock until variables are set
        self.start()

    def run(self):
        app = wx.App()
        frame = SLMframe(monitor = self.parent.monitor)
        frame.Show(True)
        self.frame = frame
        self.lock.release()
        # Start GUI main loop
        app.MainLoop()

    def start_local(self):
        self.start_orig()
        # Use lock to wait for the functions to get defined
        self.lock.acquire()