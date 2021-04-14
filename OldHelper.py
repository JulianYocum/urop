import numpy as np
from scipy.interpolate import CubicSpline


def get_pwd():
    return './'
    return "/nfs/cuore1/scratch/yocum"


def lineplanecollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return None

    t = -planeNormal.dot(rayPoint - planePoint) / ndotu

    return rayPoint + t * rayDirection


def linecubecollision(cubeCenter, cubeLength, rayDirection, rayPoint, epsilon=1e-6):

    cubeCollisions = []

    halfLength = cubeLength / 2.0

    directions = np.array([
        [0,0,halfLength], #up
        [0,halfLength,0], #front
        [halfLength,0,0], #right
    ])

    planeCollisions = []
    for i in range(6):
        if i >= 3:
            faceNormal = -directions[i%3] # to get down, back, left
        else:
            faceNormal = directions[i]

        facePoint = cubeCenter + faceNormal

        collision = lineplanecollision(faceNormal, facePoint, rayDirection, rayPoint)
        if collision is not None:
            planeCollisions.append(collision)

    #check if intersection is outside cube
    for collision in planeCollisions:

        inside = True
        for i in range(3):
            if collision[i] > (cubeCenter[i] + halfLength + epsilon) or collision[i] < (cubeCenter[i] - halfLength - epsilon):
                inside = False

        if inside:
            cubeCollisions.append(collision)

    return cubeCollisions


def channelcollisions(line, coords, epsilon=1e-6):

    rayDirection = line[3:]
    rayPoint = line[:3]

    #rayDirection = linepoints[1] - linepoints[0]
    #rayPoint = linepoints[0]
    cubeLength = 50

    #start = time.time()

    hit_channels = []
    miss_channels = []
    track_distances = []

    for channel in range(1,len(coords)+1):
        cubeCenter = coords[channel]

        #check if cubeCenter is within range of line
        CP = cubeCenter - rayPoint
        distance_to_line = np.abs(np.linalg.norm(cross(CP,rayDirection)) / np.linalg.norm(rayDirection))

        #print(channel, distance_to_line, cubeCenter, rayPoint, CP)

        if distance_to_line < cubeLength/2*np.sqrt(3) + epsilon:
        #if distance_to_line < cubeLength*np.sqrt(3) + epsilon:

            collision = linecubecollision(cubeCenter, cubeLength, rayDirection, rayPoint)
            if len(collision) == 2:
                hit_channels.append(channel)
                track_distances.append(np.linalg.norm(collision[1] - collision[0]))
            else:
                miss_channels.append(channel)

    return (hit_channels, miss_channels, track_distances)


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