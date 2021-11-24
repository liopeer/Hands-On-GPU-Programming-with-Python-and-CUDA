# unfinished

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import nupy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ker = SourceModule("""

// what position in in block + which_block*blocksize
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )

// grid measures
#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )

// 
#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )

""") # all CUDA C code comes in here

# get the "conway_ker" global function from our source module
conway_ker = ker.get_function("conway_ker")

def update_gpu(frameNum, img, newLattice_gpu, lattice_gpu, N):

    # call our kernel on specified number of blocks/grid
    # block size must be max 1024, grid size accordingly
    # kernel should now check neighbours in lattice and update newLattice
    # we then set our current image to newLattice (it is a pointer!)
    # finally we copy newLattice in (current)lattice
    conway_ker(newLattice_gpu, lattice_gpu, grid=(N/32, N/32, 1), block=(32, 32, 1))

    img.set_data(newLattice_gpu.get())

    lattice_gpu[:] = newLattice_gpu[:]
    return img

if __name__ == '__main__':
    # lattice size
    N = 128

    # random.choice gives us 1D array of length N^2, we reshape it to a lattice: NxN
    lattice = np.int32( np.random.choice([1,0], N*N, p=[0.25, 0.75]).reshape(N, N) )

    # instantiate it on the GPU memory
    lattice_gpu = gpuarray.to_gpu(lattice)

    # and another empty one to store update values
    newLattice_gpu = gpuarray.empty_like(lattice_gpu)

    fig, ax = plt.subplots()
    img = ax.imshow(lattice_gpu.get(), interpolation='nearest')

    # Almost like a for loop: call update_gpu() a specified number of times and show the animation
    ani = animation.FuncAnimation(fig, update_gpu, fargs=(img, newLattice_gpu, lattice_gpu, N, ) , interval=0, frames=1000, save_count=1000)    


    