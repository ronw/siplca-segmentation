# This is a primitive way to check for memory leaks.
# Run with ``python EatMem.py`` and use some memory watching tool
# (like ``top`` under unix) to see if the memory consumpiton goes up
# by steps while the program is running (which would suggest a memory leak).

import gc
from mlabwrap import mlab
try:
    import numpy
    from numpy.random import rand, randn
    toscalar = lambda a:a.item()
except ImportError:
    import Numeric as numpy
    from MLab import rand, randn
    toscalar = lambda a:a.toscalar()
import time
#gc.set_debug(gc.DEBUG_LEAK)
myMat = numpy.ones((1000**2,2), 'd')
mlab._set('mymat', myMat)
raw_input("Press return when ready!")
for i in range(10):
    mlab.sum(myMat,nout=0)
for i in range(1000):
    s = mlab.sin(rand(100,100))
#print loads of crap to make sure this won't screw up
for i in range(20):
    mlab.sin(rand(100,100), nout=0)
assert numpy.alltrue(myMat.flat == mlab._get('mymat',1).flat)
gc.collect()
raw_input("Press return to finish")

