import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel

host_data = np.float32( np.random.random(50000000) ) # 50million

# arguments (pointers), operation, name (why does it need one?)
gpu_2x_ker = ElementwiseKernel(
"float *in, float *out",
"out[i] = 2*in[i];",
"gpu_2x_ker")

def speedcomparison():
    t1 = time()
    host_data_2x =  host_data * np.float32(2) # on the CPU
    t2 = time()
    print('total time to compute on CPU: %f' % (t2 - t1))

    device_data = gpuarray.to_gpu(host_data) # copy original data to GPU and return pointer
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data) # create empty data structure on GPU and return pointer
    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()
    from_device = device_data_2x.get() # copy computed values back
    print('total time to compute on GPU: %f' % (t2 - t1))
    print('Is the host computation the same as the GPU computation? : {}'.format(np.allclose(from_device, host_data_2x) ))
    

if __name__ == '__main__':
    speedcomparison()
