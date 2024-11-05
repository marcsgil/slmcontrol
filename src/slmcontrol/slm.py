from numpy.typing import NDArray
import numpy as np
from juliacall import Main as jl
jl.seval("using SpatialLightModulator")


class SLMDisplay:
    """
    A class to control a Spatial Light Modulator (SLM).

    Attributes:
        monitor_id (int): The ID of the monitor to use.
        slm (jl.SLM): The SLM instance.
        width (int): The width of the SLM.
        height (int): The height of the SLM.
        refreshrate (int): The refresh rate of the SLM.
    """

    def __init__(self, monitor_id=jl.lastindex(jl.GetMonitors()) - 1) -> None:
        """
        Initialize the SLM instance.

        Args:
            monitor_id (int): The ID of the monitor to use. Defaults to the last monitor.
        """
        self.monitor_id = monitor_id
        self.slm = jl.SLMDisplay(monitor_id + 1)
        self.width = self.slm.width
        self.height = self.slm.height
        self.refreshrate = self.slm.refreshrate

    def updateArray(self, holo: NDArray[np.uint8], sleep=0.15) -> None:
        """
        Update the hologram displayed on the SLM.

        Args:
            holo: A 2D matrix of UInt8 values representing the hologram. 
                The first dimension is the width and the second dimension is the height.
        """
        jl.updateArray(self.slm, holo.T, sleep=sleep)

    def close(self) -> None:
        """
        Close the SLM window.
        """
        jl.close(self.slm)
