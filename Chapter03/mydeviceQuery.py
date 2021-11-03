import pycuda.driver as drv
drv.init()

print('Detected {} CUDA capable device(s)'.format(drv.Device.count()))