import numpy.ctypeslib as ctl
import ctypes
import numpy as np

libname = 'libcoll.so'
libdir = './'
lib=ctl.load_library(libname, libdir)

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)

channelcollisions = lib.channelcollisions
channelcollisions.argtypes = [ctl.ndpointer(np.float64, 
                                         flags='aligned, c_contiguous'),
                             c_int_p,
                             c_int_p,
                             c_double_p]

#channelcollisions(np.array([1,2,3,4,5,6], dtype=np.float64))

data1 = (ctypes.c_int * 50)()
data2 = (ctypes.c_int * 50)()
data3 = (ctypes.c_double * 50)()

res = channelcollisions(np.array([1,2,3,4,5,6], dtype=np.float64), 
                        ctypes.cast(data1, c_int_p),
                        ctypes.cast(data2, c_int_p), 
                        ctypes.cast(data3, c_double_p))

print(len(data1))
print(data2[1])
print(data3[1])


