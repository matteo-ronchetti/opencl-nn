# import numpy as np
# import pyopencl as cl
#
# a_np = np.random.rand(5000).astype(np.float32)
# b_np = np.random.rand(5000).astype(np.float32)
#
# ctx = cl.create_some_context(False)
# queue = cl.CommandQueue(ctx)
#
# mf = cl.mem_flags
# a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
# b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
#
# prg = cl.Program(ctx, """
# __kernel void sum(
#     __global const float *a_g, __global const float *b_g, __global float *res_g)
# {
#   int gid = get_global_id(0);
#   res_g[gid] = a_g[gid] + b_g[gid];
# }
# """).build()
#
# res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
# prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
#
# res_np = np.empty_like(a_np)
# cl.enqueue_copy(queue, res_np, res_g)
#
# # Check on CPU with Numpy:
# print(res_np - (a_np + b_np))
# print(np.linalg.norm(res_np - (a_np + b_np)))
import pyopencl as cl

print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')

# Print each platform on this computer
for platform in cl.get_platforms():
    print('=' * 60)
    print('Platform - Name:  ' + platform.name)
    print('Platform - Vendor:  ' + platform.vendor)
    print('Platform - Version:  ' + platform.version)
    print('Platform - Profile:  ' + platform.profile)
    # Print each device per-platform
    for device in platform.get_devices():
        print('    ' + '-' * 56)
        print('    Device - Name:  ' + device.name)
        print('    Device - Type:  ' + cl.device_type.to_string(device.type))
        print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
        print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
        print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size / 1024.0))
        print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size / 1024.0))
        print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size / 1073741824.0))
        print('    Device - Max Buffer/Image Size: {0:.0f} MB'.format(device.max_mem_alloc_size / 1048576.0))
        print('    Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
print('\n')
