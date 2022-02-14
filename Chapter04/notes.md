## Notes on Chapter 4
 - obviously LIFE/Conway's game of life can be parallelized
 - not the iterations though
 - only the single pixels in one iteration: assign 1 thread to 1 pixel, let this thread check the neighboring pixels, update the pixels value accordingly
 - IMPORTANT: all pixels have to check their neighbours and may only update their value once every thread has finished evaluating its neighbours

### First implementation
**Kernel**:  
 - __global__ functions are called by the host, __device__ functions are called by a thread on the device but are also executed on the device (I don't think the device can call functions that are implemented on the host)
 - #define is used for macros. the preprocessor will include those macros in the code and then compile
 - the grid is made up of 1. Threads, 2. Blocks (at most 1024 threads), 3. Grid

### Second implementation
 - before we let the host do the updating of the image
 - now we wanna give everything to the kernel: pass the random field and get the thing back after some iterations
 - for this we need to make sure every thread=pixel is ready with evaluating stuff before updating everything
 - and also all pixels=threads must be updated before a new neighbor-check is started
 - since we can only synchronize single blocks and not the whole grid we must constrain ourselves to a grid consisting of a single block (=max 1024 threads)

### Blocks & Grids
 - synchronizing all threads in one block can be done with the __syncthreads() function in CUDA **ONLY IN ONE SINGLE BLOCK, NOT THE WHOLE GRID!**

### Global Memory vs. Shared Memory
 - so far we've been using the GPU global memory
 - but we can also use the faster shared memory (**shared between a block of threads**)
 - shared mem. also not directly accessible by host, must be copied back first

### Shared Memory
 - declare shared array: __shared__ int a[10] // int array of length 10 that is shared among all threads in the same block
 