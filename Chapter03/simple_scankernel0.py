import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel

seq = np.array([1,2,3,4],dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32, "a+b") # this works the same way as reduce() in Python, operation specified as string in C syntax
print sum_gpu(seq_gpu).get() # this time we did not allocate space on GPU, so we get() it directly
print np.cumsum(seq)
