from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from multimethod import multimethod

class Camera(ABC):
    def __init__(self,resX,resY):
        self.resX = resX
        self.resY = resY

    @abstractmethod
    def capture(self):
        pass

    @abstractmethod
    def close(self):
        pass

class TestCamera(Camera):
    def __init__(self,resX,resY):
        super().__init__(resX,resY)
    
    @multimethod
    def capture(self, saving_path: str):
        image = Image.fromarray(np.random.randint(0,255,size=(self.resY,self.resX),dtype='uint8'))
        image.save(saving_path)

    @multimethod
    def capture(self):
        return np.random.randint(0,255,size=(self.resY,self.resX),dtype='uint8')

    def close(self):
        pass