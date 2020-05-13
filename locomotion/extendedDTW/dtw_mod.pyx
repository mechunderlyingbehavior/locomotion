## Modification of the DTW methods in the MLPY package by Soo Go

import numpy as np
cimport numpy as np
from libc.stdlib cimport * 
from cdtw_mod cimport *

np.import_array()

def dtw_met( x, y, dist_only=True ):
  print( "ack!" )
