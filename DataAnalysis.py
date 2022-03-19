import numpy as np
import pandas as pd
import scipy as sc
import uproot as up
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from scipy import stats
import os
import math
from pathlib import Path
import time
import copy
import sys
import pickle

from sklearn.metrics import r2_score

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.algorithms.unsga3 import UNSGA3
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.nsga2 import NSGA2

from pymoo.factory import get_problem, get_reference_directions

#from pymoo.util.termination.default import MultiObjectiveDefaultTermination

from Helper import *
from MyProblem import MyProblem

        
class DataAnalysis():
    def __init__(self, auto=False, load=False, eventfile='', clusterfile=''): 
        
        self.pwd = get_pwd()
        
        self.coords = load_coords(self.pwd)
        self.dEdx_pdf = make_pdf(self.pwd + '/data/pdf/dEdx/bins_dEdx_muon.csv', self.pwd + '/data/pdf/dEdx/values_dEdx_muon.csv', domain_range=[0,190])
        
        self.noisy = []
        self.dead = []
        
        self.paretoX = dict()
        self.paretoF = dict()
        
        if load:
            self.load_eventdf(eventfile)
            self.load_clusterdf(clusterfile)
            
        elif auto:
            #self.eventdf = self.load_data()
            self.load_data()
            self.load_errorchannels(self)
            #self.filter_baseline()
            self.arrange_clusters(5, 1.0) # >= 5 events, <= 1.0 seconds
            self.make_clusterdf(basicfit=True)
            self.filter_fit(3.5, 5) # <= 1.5 NRMSE, >= 0 channels
    
    def get_eventdf(self):
        return copy.deepcopy(self.eventdf)
    
    def set_eventdf(self, df):
        self.eventdf = df
        
    def set_clusterdf(self, df):
        self.clusterdf = df
        
    def load_eventdf(self, file):
        self.eventdf = pd.read_csv(file)
        self.eventdf = self.eventdf.sort_values(by=['MaxTime'])
        self.eventdf = self.eventdf.reset_index(drop=True)
        
        
    def load_clusterdf(self, file):
        self.clusterdf = pd.read_csv(file, low_memory=True, sep=',')
        self.clusterdf = self.clusterdf.sort_values(by=['Cluster'])
        self.clusterdf = self.clusterdf.reset_index(drop=True)
        self.clusterdf['Fitline'] = self.clusterdf['Fitline'].apply(
            lambda rawline: np.array(rawline.strip('[]').split() if isinstance(rawline, str) else rawline, dtype=np.float64)
        )
    
    def load_paretoX(self, file):
        with open(file, 'rb') as f:
            self.paretoX = pickle.load(f)
            
    def load_paretoF(self, file):
        with open(file, 'rb') as f:
            self.paretoF = pickle.load(f)
        
    def load_errorchannels(self):
        self.dead = np.genfromtxt(self.pwd + '/data/errorchannels/dead_channels.csv', delimiter=',')
        self.noisy = np.genfromtxt(self.pwd + '/data/errorchannels/noisy_channels.csv', delimiter=',')
        
        self.dead.astype(int)
        self.noisy.astype(int)
        
    
    def save_eventdf(self, path):
        self.eventdf.to_csv(path, index=False)
    
    def save_clusterdf(self, path):
        self.clusterdf.to_csv(path, index=False)
        
    def save_paretoX(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.paretoX, f, pickle.HIGHEST_PROTOCOL)
            
    def save_paretoF(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.paretoF, f, pickle.HIGHEST_PROTOCOL)
        
    def save_errorchannels(self):
        with open(self.pwd + "/data/errorchannels/dead_channels.csv", "a") as f:
            np.savetxt(f, self.dead, delimiter=",")
            
        with open(self.pwd + "/data/errorchannels/noisy_channels.csv", "a") as f:
            np.savetxt(f, self.noisy, delimiter=",")
        
    
    def load_data(self):
        frames = []
        path = self.pwd + '/data/ds3564/'
        
        #num_towers = len(os.listdir(path))
        num_towers = 19
        filename = 'ds3564Tower'

        for t in range(1, num_towers + 1):
            new_path = path + filename + str(t) + '.root'

            #load tower
            event = up.open(new_path)['tree']

            #recast the data as a pandas dataframe and append to frames
            frames.append(event.pandas.df())

        raw = pd.concat(frames)
        
        #adjust variable from milli to seconds
        raw['MaxPosInWindow'] = raw['MaxPosInWindow'] / 1000.0
        
        #set saturated flag
        raw['IsSaturated'] = np.logical_or((raw['Baseline'] + raw['MaxToBaseline']) > 9000, raw['SelectedEnergy'] > 25000)
        
        run_starttimes = self.runstarttimes()
        
        for run in raw['Run'].unique():
            raw.loc[raw['Run'] == run, ['Time']] += run_starttimes[run]
        
        self.eventdf = raw
        #return raw

    
    def runstarttimes(self):

        times = {}

        # fix times
        with open(self.pwd + "/data/ds3564_start_stop_times.txt") as f:
            f.readline()
            f.readline()

            first = True

            for line in f:
                linedata = line.split('|')
                linedata = [i.lstrip().rstrip() for i in linedata]

                if linedata[2] == "Background" and linedata[6] == "OK (0)":

                    linedate = datetime.strptime(linedata[3], "%b %d, %Y %H:%M:%S%z")
                    #print(linedate.timestamp())
                    timestamp = linedate.replace(tzinfo=timezone.utc).timestamp()

                    # save first run timestamp
                    if first:
                        first_timestamp = timestamp
                        first = False

                    times[int(linedata[1])] = timestamp - first_timestamp
        return times
    
    
    def eventsperchannel(self):
        # get num events per channel
        events = []
        for c in range(1,max(self.eventdf['Channel']) + 1):
            events.append(len(self.eventdf[self.eventdf['Channel'] == c]))
        return events
    
    
    # find dead channels
    def deadchannels(self):
        if len(self.dead) == 0:
            channel_events = self.eventsperchannel()

            dead = []
            for c in range(1,max(self.eventdf['Channel']) + 1):     
                if channel_events[c - 1] == 0:
                    dead.append(c)
            self.dead = dead
            
        return self.dead
        
    
    
    #detect outliers using above threshold IQR
    def noisychannels(self):
        if len(self.noisy) == 0:
            threshold = 1.5

            channel_events = self.eventsperchannel()        
            Q1, Q3 = np.percentile(channel_events, 25), np.percentile(channel_events, 75)
            IQR = Q3 - Q1

            upper_bound = Q3 + IQR * threshold
            noisy = [c for c in range(1,max(self.eventdf['Channel']) + 1) if channel_events[c - 1] > upper_bound]
            self.noisy = noisy
        
        return noisy

    
    def filter_noisy(self):
        self.eventdf = self.eventdf[np.isin(self.eventdf['Channel'], self.noisychannels(), invert=True)]
        
        return self
        
        
    def filter_baseline(self):
        self.eventdf = self.eventdf[(self.eventdf['Baseline'] + self.eventdf['MaxToBaseline']) < 9000]
        self.eventdf = self.eventdf[(self.eventdf['SelectedEnergy']) < 25000]

        return self
        
    
    def arrange_clusters(self, e_thresh=3, t_thresh=1.0):
        
        e_thresh = int(e_thresh)
        t_thresh = float(t_thresh)
        
        sorted_df = self.eventdf.copy()
        
        sorted_df['MaxTime'] =  self.eventdf[['Time', 'MaxPosInWindow']].sum(axis=1) # sort by 'Time' + 'MaxPosInWindow'
        #sorted_df['MaxTime'] =  self.eventdf[['Time', 'OFdelay']].sum(axis=1) # sort by 'Time' + 'MaxPosInWindow'
        #sorted_df['MaxTime'] = self.eventdf['Time'] # Sort by 'Time'
        
        
        sorted_df = sorted_df.sort_values(by=['MaxTime'])
        sorted_df = sorted_df.reset_index(drop=True)

        new_df = sorted_df.copy()

        #print(new_df[50:70])

        new_df['Cluster'] = [-1]*len(new_df)

        #get events that are clustered
        row = 0
        events = 1
        cluster = [row]
        cluster_num = 0

        while (row < len(self.eventdf)):

            #make sure there is a next event. if at end of dataframe, set times to fail next test
            if(row < len(self.eventdf) - 1):
                successive_time = sorted_df.iloc[row + 1]['MaxTime'] #+ sorted_df.iloc[row + 1]['MaxPosInWindow']/1000.0
                event_time = sorted_df.iloc[row]['MaxTime'] #+ sorted_df.iloc[row]['MaxPosInWindow']/1000.0
            else:
                event_time = 0
                successive_time = t_thresh + 1


            if abs(successive_time - event_time) <= t_thresh:
                events += 1
                cluster.append(row + 1)
            else:   

                #print(events)
                if events < e_thresh:
                    for i in cluster:
                        new_df = new_df.drop(i) #sorted_df.index[i])
                else:
                    #clusters.append(cluster)
                    for i in cluster:
                        #print(cluster_num)
                        new_df.loc[i, 'Cluster'] = cluster_num
                    cluster_num += 1

                events = 1
                cluster = [row + 1]

            row += 1
        
        self.eventdf = new_df
    
        return self


    # returns array of 3 arrays corresponding to x y z
    def clustercoords(self, cluster):

        #coords = []
        x = []
        y = []
        z = []

        for c in cluster['Channel']:
            #coords.append([ch_coords[c][0], ch_coords[c][1], ch_coords[c][2]])
            x.append(self.coords[c][0])
            y.append(self.coords[c][1])
            z.append(self.coords[c][2])

        return [x,y,z]

    
    # takes dataframe of a single cluster and finds line of best fit
    def basicfit(self, cluster):
        coords = self.clustercoords(cluster)

        data  = np.array(coords).T

        datamean = data.mean(axis=0)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data - datamean)

        #linepts = vv[0] * np.mgrid[-400:400:2j][:, np.newaxis]

        # shift by the mean to get the line in the right place
        #linepts += datamean
        
        v = vv[0] / np.linalg.norm(vv[0])
        p = datamean

        #return linepts
        return np.round(np.append(p, v), decimals=6)
    
    
    
#     def initial_sample(self, hit_chs, N):
    
#         samples = []
#         sample_chs = np.unique(hit_chs)
#         for i in range(N):

#             ch_a, ch_b = np.random.choice(sample_chs, 2, replace=False)

#             pt_a = self.coords[ch_a]
#             pt_b = self.coords[ch_b]

#             line_pts = np.concatenate((pt_a, pt_b))

#             samples.append(line_pts)

#         return np.array(samples)
    
    
    # takes dataframe of a single cluster and finds line of best fit
    def fitline(self, cluster, pop_num=50, gen_num=200, verbose=False, save_pareto=False):
        cluster_num = cluster['Cluster'].array[0]
        
        hit_chs = cluster['Channel'].values
        hit_chs = hit_chs[np.isin(hit_chs, self.noisy, invert=True)]

        non_sat_chs = cluster[cluster['IsSaturated'] == False]['Channel'].values
        non_sat_chs = non_sat_chs[np.isin(non_sat_chs, self.noisy, invert=True)]
        
        ### HANDLE THIS MORE EXPLICITLY
        if len(hit_chs) == 1:
            return self.basicfit(cluster)
        
        miss_chs = np.array([ch for ch in range(1,989) if ch not in hit_chs])
        miss_chs[np.isin(miss_chs, self.dead, invert=True)]
        
        
        
#         initial_sample =  350 * np.sqrt(3) * np.array([[-1,0,0,1,0,0],
#                                                        [0,-1,0,0,1,0],
#                                                        [0,0,-1,0,0,1]])
        
               
        PRELIM_POP = 200
        PRELIM_GENS = 200
        
        #prelim_ref_dirs = get_reference_directions("energy", 2, PRELIM_POP) #, seed=1)
            
        prelim_algorithm = NSGA2( #UNSGA3
            pop_size=PRELIM_POP,
            #ref_dirs=prelim_ref_dirs,
            n_offsprings=PRELIM_POP,
            sampling=get_sampling("real_random"), #self.initial_sample(hit_chs,PRELIM_POP), #
            crossover=get_crossover("real_sbx", prob=.9, eta=15), #eta=15
            mutation=get_mutation("real_pm", eta=20), #eta=20
            eliminate_duplicates=True
        )
        
        
        prelim_problem = MyProblem(hit_chs, miss_chs, non_sat_chs, cluster['SelectedEnergy'].values, num_obj=2)
        
        prelim_result = minimize(prelim_problem,
                       prelim_algorithm,
                       get_termination("n_gen", PRELIM_GENS),
                       #seed=1
                       #pf=problem.pareto_front(use_cache=False),
                       #save_history=True)
                       verbose=verbose
                      )
        
#         if verbose:
#             print("prelim")
#             print(prelim_result.X)
#             print(prelim_result.F)
        
#         if verbose:

#             pre_bestline = []
#             pre_best_missed = np.inf
#             pre_best_extra = np.inf

#             for i in range(len(prelim_result.X)):

#                 betterline = pts_to_line(prelim_result.X[i]) #index])
#                 hit_channels = channelcollisions(betterline, self.coords)[0]
#                 missed, extra = self.errorchannels(cluster, hit_channels)
#                 #linear = prelim_result.F[i][2]

#                 #print(len(missed), len(extra))#, linear)

#                 if len(missed) < pre_best_missed:
#                     pre_best_missed = len(missed)
#                     pre_best_extra = len(extra)
#                     pre_bestline = betterline

#                 elif (len(missed) == pre_best_missed) and (len(extra) < pre_best_extra):
#                     pre_best_extra = len(extra)
#                     pre_bestline = betterline

#             print("prelim best: ", pre_bestline, pre_best_missed, pre_best_extra)
        
        ########################################
        if gen_num is not None and pop_num is not None:
        
            final_ref_dirs = get_reference_directions("energy", 3, pop_num) #, seed=1)

            final_algorithm = UNSGA3( #NSGA3
                pop_size=pop_num,
                ref_dirs=final_ref_dirs,
                sampling=prelim_result.X,
                #n_offsprings=pop_num,
                crossover=get_crossover("real_sbx", prob=.9, eta=15), #eta=15
                mutation=get_mutation("real_pm", eta=20), #eta=20
                eliminate_duplicates=True
            )
        
            # set constraint for top ~5% of prelim scores
            constraint = np.sort(prelim_result.F[:,0] + prelim_result.F[:,1])[80] #[round(len(prelim_result.F)/20)]
            #print("constraint is: ", constraint)

            final_problem = MyProblem(hit_chs, miss_chs, non_sat_chs, cluster['SelectedEnergy'].values, num_obj=3, constraint=constraint)

            final_result = minimize(final_problem,
                           final_algorithm,
                           get_termination("n_gen", gen_num),
                           verbose=verbose
                          )

            if verbose:
                print("final")
                #print(final_result.X)
                #print(final_result.F)
        
        else:
            final_result = None
        
        bestline = []
        
        best_missed = np.inf
        best_extra = np.inf
        best_linear = np.inf
        
        #         f = 0 * res.F[:,0] + 0 * res.F[:,1] + res.F[:,2] # possibly add weights here?
#         sorted_f = sorted(f)
        
        bestline = [] #res.X[np.where(f==sorted_f[0])[0][0]]

        #find best line out of top ten
        if not hasattr(final_result, 'X') or final_result.X is None:
            print("Many-Objective Failure, defaulting to Multi-Objective...")
            
            final_result = prelim_result
            final_result.F = np.insert(prelim_result.F, 2, np.inf, axis=1)
        
        for i in range(len(final_result.X)):
        #for betterline in res.X:

            #index = np.where(f==sorted_f[i])[0][0]
            
            #print(final_result.X[i])

            betterline = pts_to_line(final_result.X[i]) #index])
            hit_channels = channelcollisions(betterline, self.coords)[0]
            missed, extra = self.errorchannels(cluster, hit_channels)
            linear = final_result.F[i][2]   #index][2]         
           
            if verbose:
                print(len(missed), len(extra), linear)

            if len(missed) < best_missed:
                best_missed = len(missed)
                best_extra = len(extra)
                best_linear = linear
                bestline = betterline

            elif (len(missed) == best_missed) and (len(extra) < best_extra):
                best_extra = len(extra)
                best_linear = linear
                bestline = betterline
                
            elif (len(missed) == best_missed) and (len(extra) == best_extra) and (linear < best_linear):
                best_linear = linear
                bestline = betterline
        
        if verbose:
            print("final best: ", bestline, best_missed, best_extra, best_linear)
            
        if save_pareto:
            self.paretoX[cluster_num] = final_result.X
            self.paretoF[cluster_num] = final_result.F
    
        return np.round(bestline, decimals=6) #, prelim_result, final_result
    
        
    def dEdx(self, cluster, line, show_graph=False):

        hit_channels, _, track_distances = channelcollisions(line, self.coords)
        
        dEdxs = []
                    
        for i, row in cluster.iterrows():
            #non_sat_chs.append(row['Channel'])
            if row['Channel'] in hit_channels:
                #energies.append(row['SelectedEnergy'])
                energy = row['SelectedEnergy']
                #track_distance = track_distances[np.where(hit_channels == row['Channel'])[0][0]]
                track_distance = track_distances[hit_channels.index(row['Channel'])]
                dEdxs.append(energy / track_distance)
            else:
                dEdxs.append(np.nan)
        
        return np.array(dEdxs)
    
    def likelihood(self, dEdxs):
        p_densities = self.dEdx_pdf(dEdxs[~(np.isnan(dEdxs))]/100)
        log_densities = np.log(p_densities)
        
        return log_densities.sum()
    
    
    def NRMSE(self, cluster, line):
        ''' gets NRMSE for a given cluster
            use distance from point to line of best fit as residual where
            d = |(p-a)x(p-b)|/|b-a|
            and variables are vectors'''
        
        if len(cluster) <= 2:
            return 0

        dlist = []
        
        # store 2 best fit lines as vectors
        linepoints = line_to_pts(line)
        a = linepoints[0]
        b = linepoints[1]

        for index, event in cluster.iterrows():

            p = np.array(self.coords[event['Channel']])
            d = np.linalg.norm(cross(p-a, p-b)) / np.linalg.norm(b-a)

            dlist.append(d)

        # get root mean squared error for cluster
        RMSE = math.sqrt(sum([i**2 for i in dlist])/(2* len(dlist) - 4))

        # normalize
        NRMSE = RMSE / 4.54**2

        return NRMSE
     
        
    def errorchannels(self,cluster, hitchannels):
        
        #hitchannels = self.channelcollisions(line)[0]
        clusterchannels = cluster['Channel'].unique()
        
        if -1 not in clusterchannels:
        
            extra = [ch for ch in hitchannels if ch not in clusterchannels]
            missing = [ch for ch in clusterchannels if ch not in hitchannels]

            return (missing, extra)
        
        else:
            return ([], [])
        
        
    def zenith(self, line):
        
        linepoints = line_to_pts(line)

        z = abs(linepoints[0][2] - linepoints[1][2])
        d = abs(np.linalg.norm(linepoints[0] - linepoints[1]))
        return math.acos(z/d)
    
    
    def azimuth(self, line):
        
        linepoints = line_to_pts(line)

        y = linepoints[0][1] - linepoints[1][1]
        x = linepoints[0][0] - linepoints[1][0]
        
        # confine to 1st and 4th quadrant
        if x < 0:
            y*=-1
            x*=-1
        
        az = math.atan2(y,x)
        az -= 36.24 * np.pi / 180
        
        if az <= -np.pi/2:
            az += np.pi
        
        return az
       
    
    def make_clusterdf(self, pop_num=None, gen_num=None, basicfit = False, verbose=False, save_pareto=False):
        # get clusters
        clusters = np.unique(self.eventdf['Cluster'])
        
        eventspercluster = []
        channelspercluster = []    
        starttimes = []
        timespreads = []
        fitlines = []
        NRMSE = []
        extrachannels = []
        missingchannels = []
        zeniths = []
        azimuths = []
        likelihoods = []
        #dEdxs = []
        #dEdx_errs = []
        
        #add columns to eventdf
        
        self.eventdf['Hit'] = False
        self.eventdf['PathLength'] = np.nan
        self.eventdf['dEdx'] = np.nan
        
        for c in clusters:

            cluster = self.eventdf[self.eventdf['Cluster'] == c]

#             #event and channel info
#             if len(cluster) == 1 and cluster['Channel'].values[0] == -1:
#                 eventspercluster.append(0)
#                 channelspercluster.append(0)
#             else:
            eventspercluster.append(len(cluster))
            channelspercluster.append(len(cluster['Channel'].unique()))

            #get timespread
            clustertimes = cluster['MaxTime']
            starttimes.append(min(clustertimes))
            timespreads.append(max(clustertimes) - min(clustertimes))

            #get fitline
            if basicfit:
                fitline = self.basicfit(cluster)
#             elif pop_num is not None and gen_num is not None:
#                 fitline = self.fitline(cluster, pop_num, gen_num, verbose)
#            elif pop_num is None and gen_num is None:
#                fitline = self.fitline(cluster)
#             else:
#                 print("Error: fitline unspecficied")
#                 sys.exit()
            else:
                fitline = self.fitline(cluster, pop_num, gen_num, verbose, save_pareto)
                
            fitlines.append(fitline)
            
            #get NRMSE
            NRMSE.append(self.NRMSE(cluster, fitline))
                        
            #get missing and extra channels
            hit_channels, _, track_distances = channelcollisions(fitline, self.coords)
            missing, extra = self.errorchannels(cluster, hit_channels)
            
            #save path length data to eventdf
            hit_data = [ch in hit_channels for ch in cluster['Channel'].values]
            path_data = []
            for i in range(len(hit_data)):
                if hit_data[i]:
                    index = hit_channels.index(cluster['Channel'].values[i])
                    path_data.append(track_distances[index])
                else:
                    path_data.append(np.nan)
                    
            self.eventdf.loc[self.eventdf['Cluster'] == c, "Hit"] = hit_data
            self.eventdf.loc[self.eventdf['Cluster'] == c, "PathLength"] = path_data  
            
            #cluster['PathLength'] = track_distances
            
            extrachannels.append(len(extra))
            missingchannels.append(len(missing))
            
            #get angles
            zeniths.append(self.zenith(fitline))
            azimuths.append(self.azimuth(fitline))
                        
            #get energies
            dEdxs = self.dEdx(cluster, fitline)
            self.eventdf.loc[self.eventdf['Cluster'] == c, "dEdx"] = dEdxs
            
            #get likelihood
            likelihoods.append(self.likelihood(dEdxs))
            
            #dEdx, err = self.dEdx(cluster, fitline)
            #dEdxs.append(dEdx)
            #dEdx_errs.append(err)
        
        #zeniths_degrees = [theta*360/(2*math.pi) for theta in zeniths]
        #cos_theta = [math.cos(theta) for theta in zeniths]

        d = {'Cluster' : clusters, 'Events' : eventspercluster, 'Channels' : channelspercluster, \
            'StartTime': starttimes, 'TimeSpread' : timespreads, 'NRMSE' : NRMSE, \
             'Zenith' : zeniths,'Azimuth': azimuths, 'ExtraCh': extrachannels, 'MissingCh' : missingchannels, \
             'Likelihood' : likelihoods, 'Fitline' : fitlines}
             #'dEdx': dEdxs, 'dEdx_err': dEdx_errs, 'Fitline' : fitlines}

        #return newdf
        self.clusterdf = pd.DataFrame(data=d)
        
        return self
        
        
    def filter_fit(self, NRMSE=None, channels=None):
        if NRMSE:
            self.clusterdf = self.clusterdf[self.clusterdf['NRMSE'] < NRMSE]
        if channels:
            self.clusterdf = self.clusterdf[self.clusterdf['Channels'] >= channels]
        
        if hasattr(self, 'eventdf'):
            self.eventdf = self.eventdf[self.eventdf['Cluster'].isin(self.clusterdf['Cluster'].values)]
        
        return self


    def get_clusterdf(self, fitline=True):
        if fitline:
            return copy.deepcopy(self.clusterdf)
        return copy.deepcopy(self.clusterdf).iloc[:,:-1]
    
    
    def get_clusterrate(self):
        
        num_clusters = len(self.clusterdf)

        # fix times
        with open(self.pwd + "/data/ds3564_start_stop_times.txt") as f:
            f.readline()
            f.readline()

            first = True

            sum = timedelta()

            for line in f:
                linedata = line.split('|')
                linedata = [i.lstrip().rstrip() for i in linedata]

                if linedata[2] == "Background" and linedata[6] == "OK (0)":

                    (h, m, s) = linedata[5].split(':')
                    d = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
                    sum += d

        total_seconds = sum.total_seconds()
        
        print('clusters: ' + str(num_clusters))
        print('run time: ' + str(total_seconds))

        return float(num_clusters) / float(total_seconds)
    
    
    def get_cluster(self, cluster_list):
        if isinstance(cluster_list, (int, np.int64)):
            cluster_list = [cluster_list]
        
        return copy.deepcopy(self.eventdf[self.eventdf['Cluster'].isin(cluster_list)])
    
    
    def get_fitline(self, cluster_num):
            
        return self.clusterdf[self.clusterdf['Cluster'] == cluster_num]['Fitline'].values[0]
    
    
    def plot_linear_cube(self, ax, X, Y, Z, dx=50, dy=50, dz=50, color='blue'):
#     fig = plt.figure(figsize=(15,15))
#     ax = Axes3D(fig)

        print(X,Y,Z)
    
        for x,y,z in zip(X,Y,Z):
            #print(x,y,z)
            x,y,z = x-dx/2, y-dy/2, z-dz/2
            xx = [x, x, x+dx, x+dx, x]
            yy = [y, y+dy, y+dy, y, y]
            kwargs = {'alpha': 1, 'color': color, 'linewidth' :.6}
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
            #plt.title('Cube')
            #plt.show()
    
    
    def show_channel(self, channel_list, x1=15, x2=45, show_crystals=False):
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_proj_type('ortho')
        
        if isinstance(channel_list, (int, np.int64)):
            channel_list = [channel_list]
            
        coords = np.array([self.coords[ch] for ch in channel_list]).T
        ax.scatter3D(*coords)
        
        if show_crystals:
            self.plot_linear_cube(ax, *coords)
        
        plt.xlim([-350,350])
        plt.ylim([-350,350])
        ax.set_zlim([-350,350])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(x1, x2)
        
        plt.show()
        
        
    def show_cluster(self, cluster_list, orientation=(15,45), show_crystals=False):
        
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_proj_type('ortho')
        
        #if given int, make list
        if isinstance(cluster_list, (int, np.int64)):
            cluster_list = [cluster_list]
            
        for c in cluster_list:
            cluster = self.eventdf[self.eventdf['Cluster'] == c]
            
            hit_cluster = cluster[cluster['Hit'] == True]
            miss_cluster = cluster[cluster['Hit'] == False]

            hit_coords = self.clustercoords(hit_cluster)
            miss_coords = self.clustercoords(miss_cluster)
            
            if show_crystals:
                self.plot_linear_cube(ax, *hit_coords, color='blue')
                self.plot_linear_cube(ax, *miss_coords, color='red')
            else:
                ax.scatter3D(*hit_coords, color='blue')
                ax.scatter3D(*miss_coords, color='red')

            #line = self.fitline(cluster)
            line = self.clusterdf[self.clusterdf['Cluster'] == c]['Fitline'].values[0]
            linepts = line_to_pts(line)
            
            ax.plot3D(*linepts.T)
            
            


        plt.xlim([-350,350])
        plt.ylim([-350,350])
        ax.set_zlim([-350,350])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(*orientation)
        
        plt.show()
        
    def show_simulation(self, cluster_num, orientation=(15, 45)):
        
        #linepoints = self.fitline(self.get_cluster(cluster_num))
        line = self.clusterdf[self.clusterdf['Cluster'] == cluster_num]['Fitline'].values[0]
        linepts = line_to_pts(line)
        
        hit_channels = channelcollisions(line, self.coords)[0]
        hit_channel_coords = np.array([self.coords[channel] for channel in hit_channels])


        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_proj_type('ortho')

        ax.scatter3D(*hit_channel_coords.T)
        ax.plot3D(*linepts.T)

        plt.xlim([-350,350])
        plt.ylim([-350,350])
        ax.set_zlim([-350,350])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(*orientation)

        plt.show()
        