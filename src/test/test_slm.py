from slmcontrol import SLMdisplay
import numpy as np


def test_SLMdisplay():
    slm = SLMdisplay(monitor=0)
    try:
        slm.updateArray(np.ones((100, 100)))
    finally:
        slm.close()
