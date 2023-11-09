from slmcontrol.structures import *
from slmcontrol.hologram import *
from slmcontrol.zernike import *

try:
    from slmcontrol.slm import SLMdisplay
except:
    raise ModuleNotFoundError("""wxPython doesn't seem to be installed! 
If you are running Linux, you must manually install it. 
Check https://wxpython.org/pages/downloads/ for more informations.
          """)