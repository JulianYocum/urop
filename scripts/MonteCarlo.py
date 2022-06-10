import random
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from DataAnalysis import *
from Helper import *
#%run DataAnalysis.ipynb

XBOUND = 323.7
YBOUND = 323.4
ZBOUND = 348.0

TIMESPAN = 2760000

class MonteCarlo(DataAnalysis):
    def __init__(self, num_tracks=None, track_type=None, particle_type="muon", allow_empty_tracks=False, flux_square_size=12):
        
        super().__init__()
        
        self.allow_empty_tracks = allow_empty_tracks
        self.flux_square_size = flux_square_size

        if num_tracks or track_type:
            self.generate_tracks(num_tracks, track_type)
            self.generate_eventdf(particle_type)
            self.arrange_clusters(1, .02)
            self.generate_clusterdf()


    def get_tracks(self):
        return self.tracks
    
    
    def make_inverse_cdf(self, angles, bin_width):
    
        cdf_values = []
        for i in range(len(angles)):
            x = angles[:,0][i] + bin_width/2
            y = 0
            for j in range(i+1):
                y += angles[:,1][j] * bin_width
            cdf_values.append((x,y))
        
        cdf_values.insert(0, [0,0])
        #cdf_values.append([1,1])
        cdf_values = np.array(cdf_values)
        #cdf_values[0][1] = 0
        #cdf_values[-1][1] = 1
        
        #print(cdf_values)

        return CubicSpline(cdf_values[:,1], cdf_values[:,0])
    
    
    def initialize_sasso(self):
        cos_theta=pd.read_csv(self.pwd + '/data/angular/cos_theta.csv', sep=',',header=None).values
        phi=pd.read_csv(self.pwd + '/data/angular/phi.csv', sep=',',header=None).values

        cos_theta = cos_theta[::-1]

        #normalize plots
        cos_theta[:,1] /= np.sum(cos_theta[:,1] / (38+1))
        phi[:,1] /= np.sum(phi[:,1] * 360 / (36+1))
        
        self.inverse_cdf_theta = self.make_inverse_cdf(cos_theta, 1/(38+1))
        self.inverse_cdf_phi = self.make_inverse_cdf(phi, 360 / (36+1))
        
    
    def sasso_track(self):
        
        theta = np.arccos(self.inverse_cdf_theta(random.random()))
        phi = - self.inverse_cdf_phi(random.random())
        
        phi = (phi + 36.24) * np.pi / 180
        
        z = np.cos(theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        
        v = np.array([x,y,z])
        p = np.array([random.uniform(-350,350) for i in range(3)])
        
        
        return np.round(np.append(p, v / np.linalg.norm(v)), decimals=6)
        

    def box_track(self):
        linepoints = np.array([[random.uniform(-350,350) for i in range(3)] for j in range(2)])

        v = linepoints[1] - linepoints[0]
        v = v / np.linalg.norm(v)

        return np.round(np.append(linepoints[0], v), decimals=6)

        #return [[random.uniform(-XBOUND,XBOUND),random.uniform(-YBOUND,YBOUND),\
        #         random.uniform(-ZBOUND,ZBOUND)] for i in range(2)]

    def flux_track(self):

        square_size = self.flux_square_size*1000 #10m

        p = np.array([random.uniform(-square_size/2, square_size/2) for i in range(2)] + [0])

        v = np.array([random.gauss(0, 1) for i in range(3)])
        v / np.linalg.norm(v)

        #b_mag = (b[0]**2 + b[1]**2 + b[2]**2)**.5

        return np.round(np.append(p, v / np.linalg.norm(v)),decimals=6)

        #return [[a[i] - 400*b[i]/b_mag for i in range(3)], [a[i] + 400*b[i]/b_mag for i in range(3)]]

    def iso_track(self):

        p = np.array([random.uniform(-350,350) for i in range(3)])
        #a = [random.uniform(-XBOUND,XBOUND),random.uniform(-YBOUND,YBOUND),\
        #         random.uniform(-ZBOUND,ZBOUND)]

        v = np.array([random.gauss(0, 1) for i in range(3)])
        #b_mag = (b[0]**2 + b[1]**2 + b[2]**2)**.5

        return np.round(np.append(p, v / np.linalg.norm(v)),decimals=6)

        #return [[a[i] - 400*b[i]/b_mag for i in range(3)], [a[i] + 400*b[i]/b_mag for i in range(3)]]


    def generate_tracks(self, num=100, track_type='iso'):

        #tracks = []
        
        if track_type == "box":
            track_gen = self.box_track

        elif track_type in ['isotropic', 'iso']:
            track_gen = self.iso_track

        elif track_type == "flux":
            track_gen = self.flux_track

        elif track_type == "sasso":
            self.initialize_sasso()
            track_gen = self.sasso_track
        else:
            print("Defaulting to isotropic...")
            track_gen = self.iso_track

#         for i in range(num):

#             tracks.append(np.array(track))

        self.tracks = [track_gen() for i in range(num)]



    def generate_eventdf(self, particle_type):

        events = []

        time_bound = TIMESPAN*len(self.tracks)/250
        rand_times = np.sort(np.random.choice(np.arange(time_bound),len(self.tracks),replace=False)).astype(np.float64)
        
        for i in range(len(self.tracks)):

            track = self.tracks[i]
            channels, _, distances = channelcollisions(track, self.coords)
            
            if particle_type == "muon":
                
                # value generated from geant4 muon pdf distribution
                bins_file = self.pwd + '/data/pdf/dEdx/bins_dEdx_muon.csv'
                values_file = self.pwd + '/data/pdf/dEdx/values_dEdx_muon.csv'
                dEdx_generator = make_inverse_cdf(bins_file, values_file)
                dEdxs = dEdx_generator(np.random.random(len(channels))) * 100
                
            else:
                
                # all channels have same random value between 0 and 1000
                dEdxs = 1000 * np.ones(len(channels)) * np.random.random()
            
            if len(channels) > 0:
                for j in range(len(channels)):
                    energy = dEdxs[j] * distances[j]
                    row = [0, int(channels[j]), rand_times[i] + j*.01, 1, 3.2, 3.2, energy, 0.0, 0.0, 0, False, i, 0, i, True, distances[j]]
                    events.append(row)
            elif len(channels) == 0 and self.allow_empty_tracks:
                row = [0, -1, rand_times[i], 1, 3.2, 3.2, 0, 0.0, 0.0, 0, False, i, 0, -1, True, 0]
                events.append(row)

        events = np.array(events)
        columns = ['Run', 'Channel', 'Time', 'NumPulses', 'OFdelay', 'MaxPosInWindow', 'SelectedEnergy',\
                  'Baseline', 'MaxToBaseline', 'StabAmp', 'IsSaturated', 'Track', 'MaxTime', 'Cluster', 'Hit', 'PathLength']

        if len(events) == 0:
            df = pd.DataFrame([], columns=columns)
        else:
            df = pd.DataFrame(events, columns=columns)

        #convert columns to ints
        df[["Run", "Channel", "NumPulses", "Track", "Cluster"]] = df[["Run", "Channel", "NumPulses", "Track", "Cluster"]].astype(int)
        df[["IsSaturated", "Hit"]] = df[["IsSaturated", "Hit"]].astype(bool)
        
        self.eventdf = df


    def generate_clusterdf(self):

        # overide DataAnalysis with generated track
        self.basicfit = lambda cluster: self.tracks[cluster['Track'].unique()[0]]
                                    
        self.make_clusterdf(basicfit=True)
