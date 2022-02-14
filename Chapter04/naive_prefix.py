import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time
import numpy as np

# this is a naive parallel prefix-sum kernel that uses shared memory
naive_ker = SourceModule("""
__global__ void naive_prefix(double *vec, double *out)
{
     __shared__ double sum_buf[1024]; // we'll use this as a buffer for our calculation
     int tid = threadIdx.x;     
     sum_buf[tid] = vec[tid]; // each thread can copy its own element to the buffer
     
     // begin parallel prefix sum algorithm
     
     int iter = 1;
     for (int i=0; i < 10; i++)
     {
         __syncthreads();
         if (tid >= iter ) // iter is similar to 2^i
         {
             sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter];            
         }
         
         iter *= 2;
     }
         
    __syncthreads();
    out[tid] = sum_buf[tid];
    __syncthreads();
        
}
""")
naive_gpu = naive_ker.get_function("naive_prefix")
    


if __name__ == '__main__':
    
    
    testvec = np.random.randn(1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)
    
    outvec_gpu = gpuarray.empty_like(testvec_gpu)

    naive_gpu( testvec_gpu , outvec_gpu, block=(1024,1,1), grid=(1,1,1))
    
    total_sum = sum( testvec)
    total_sum_gpu = outvec_gpu[-1].get() # last element of the additive sequence
    
    print("Does our kernel work correctly? : {}".format(np.allclose(total_sum_gpu , total_sum) ))
