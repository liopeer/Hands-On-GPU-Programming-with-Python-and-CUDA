import pycuda.driver as drv
drv.init()

for i in range(drv.Device.count()):
    gpu_device = drv.Device(i)
    print('Device {}: {}'.format(i, gpu_device.name()))
    compute_capability = float('%d.%d' %gpu_device.compute_capability())
    print('\t Total Memory: {} megabytes'.format(gpu_device.total_memory()//(1024**2)))