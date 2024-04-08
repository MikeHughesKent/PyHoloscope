INLINE = 1
OFF_AXIS = 2

INLINE_MODE = 1   # deprecated, kept for backwards compatibility
OFFAXIS_MODE = 2  # deprecated, kept for backwards compatibility

from pyholoscope.general import *
from pyholoscope.off_axis import *
from pyholoscope.utils import *
from pyholoscope.prop_lut import *
from pyholoscope.focus_stack import *
from pyholoscope.roi import *
from pyholoscope.holo_class import *
from pyholoscope.focusing import *
try:
    from pyholoscope.focusing_numba import *
except:
    pass
from pyholoscope.focus_shift import *

