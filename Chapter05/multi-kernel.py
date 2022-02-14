import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

# specified number of arrays of specified length ~1million
# we want to launch them all and be modified concurrently, parellelized over individual arrays
num_arrays = 200
array_len = 1024**2

# the kernel operates on one array, but we launch the same kernel for all the arrays that we have
ker = SourceModule("""       
__global__ void mult_ker(float * array, int array_len)
{
     int thd = blockIdx.x*blockDim.x + threadIdx.x; // variable for TOTAL thread index
     int num_iters = array_len / blockDim.x; // how many elements we should give to one thread

     for(int j=0; j < num_iters; j++)
     {
         int i = j * blockDim.x + thd;

         for(int k = 0; k < 50; k++)
         {
              array[i] *= 2.0;
              array[i] /= 2.0;
         }
     }

}
""")

mult_ker = ker.get_function('mult_ker')

data = []
data_gpu = []
gpu_out = []

# generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

t_start = time()

# copy arrays to GPU.
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu(data[k]))

# process arrays = launch the kernel on all arrays
for k in range(num_arrays):
    mult_ker(data_gpu[k], np.int32(array_len), block=(64,1,1), grid=(1,1,1)) 
    # blockDim.x will be 64

# copy arrays from GPU.
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get())

t_end = time()

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

print('Total time: %f' % (t_end - t_start))
