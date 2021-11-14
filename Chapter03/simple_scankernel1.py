import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel

# find maximum value in float32 array

seq = np.array([1,100,-3,-10000, 4, 10000, 66, 14, 21],dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b") # meaning: ? is the ternary operator
# syntax: condition ??? return if true ::: return if false
print(max_gpu(seq_gpu).get()[-1])
print(np.max(seq))
