import os
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

import juliapkg

juliapkg.require_julia("1.10")
juliapkg.add("StructuredLight", "13204c95-a6e5-4c09-8c7b-ee09b09e0944", version="0.6")
juliapkg.add("SpatialLightModulator", "8496614f-bc0d-4828-b8a6-7044d7be1234", version="0.3")

from slmcontrol.slm import *
from slmcontrol.hologram import *
from slmcontrol.structures import *
from slmcontrol.zernike import *