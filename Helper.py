import numpy as np
from scipy.interpolate import CubicSpline
import numpy.ctypeslib as ctl
import ctypes


def get_pwd():
    return './'
    return "/nfs/cuore1/scratch/yocum"

def channelcollisions(line, file=None):
    libname = 'libcoll.so'
    libdir = get_pwd()
    
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

    data1 = (ctypes.c_int * 30)()
    data2 = (ctypes.c_int * 30)()
    data3 = (ctypes.c_double * 30)()

    res = channelcollisions(line.astype(np.float64), 
                            ctypes.cast(data1, c_int_p),
                            ctypes.cast(data2, c_int_p), 
                            ctypes.cast(data3, c_double_p))

    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)

    hit_channels = data1[data1!=0]
    miss_channels = data2[data2!=0]
    track_lengths = data3[data3!=0]
    
    return (list(hit_channels), list(miss_channels), track_lengths)
    


def pts_to_line(line_pts):

    p = np.array(line_pts)[:3]
    a = np.array(line_pts)[3:]

    v = (a - p) / np.linalg.norm(a - p)

    x = np.array([p, v]).flatten()

    return x


def line_to_pts(line):
    x = np.array(line)

    p = x[:3]
    v = x[3:]
    return v * np.mgrid[-800:800:2j][:, np.newaxis] + p


 # create dictionary mapping channel numbers to a tuple containing coordinates (x,y,z)
def load_coords(pwd):

    coords = {}

    with open(pwd + "/data/detector_positions.txt", 'r') as f:
        for line in f:
            data = line.split(',')

            if int(data[0]) < 1000:
                coords[int(data[0])] = (float(data[1]), float(data[2]), float(data[3]))

    return coords

def make_pdf(bins_file, values_file, domain_range=None):

    bins = np.genfromtxt(bins_file, delimiter=',')
    domain = (bins[0:-1] + bins[1:]) / 2
    values = np.genfromtxt(values_file, delimiter=',')
    values = values / sum(values)

    #print(domain, values)
    
    spline = CubicSpline(domain, values)
    
    if domain_range:
        def pdf(x):
            result = spline(x)
            result[(x<domain_range[0]) | (x>domain_range[1])] = 1e-12
            return result
        return pdf
    
    return spline

def make_inverse_cdf(bins_file, values_file):

    bins = np.genfromtxt(bins_file, delimiter=',')
    domain = (bins[0:-1] + bins[1:]) / 2
    pdf_values = np.genfromtxt(values_file, delimiter=',')
    pdf_values = pdf_values / sum(pdf_values)

    cdf_values = pdf_values * 0
    for i in range(len(pdf_values)):
        cdf_values[i] = pdf_values[:i+1].sum()

    #bin_width = bins[1] - bins[0]

    spline = CubicSpline(cdf_values, domain)
    
    return spline


def ptsfromline(pts, linepts):

    a = linepts[np.newaxis].T[:3]
    b = linepts[np.newaxis].T[3:]

    d = np.linalg.norm(np.cross(pts - a, pts-b, axis=0), axis=0) / np.linalg.norm(b-a, axis=0)

    return d

    
# manually do crossproduct to avoid numpy overhead for small vectors
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
    return c