import gc
import pymat
ses = pymat.open()
from nummat import matrix
import time
gc.set_debug(gc.DEBUG_LEAK)
myMat = matrix.ones((1000**2, 2),'d')
for i in range(5):
    pymat.put(ses, 'foo', myMat.me)
