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

from sklearn.metrics import r2_score

#from pymoo.algorithms.unsga3 import UNSGA3
#from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize


from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_reference_directions

from Helper import *
from oldMyProblem import *

        
class DataAnalysis():
    def __init__(self, auto=False, load=False, eventfile='', clusterfile=''): 
        
        self.pwd = "."
        #self.pwd = str(Path().absolute())
        #self.pwd = "/nfs/cuore1/scratch/yocum"
        self.coords = load_coords(self.pwd)
        
        self.noisy = []
        self.dead = []
        
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
        
        
    def load_errorchannels(self):
        self.dead = np.genfromtxt(self.pwd + '/data/errorchannels/dead_channels.csv', delimiter=',')
        self.noisy = np.genfromtxt(self.pwd + '/data/errorchannels/noisy_channels.csv', delimiter=',')
        
        self.dead.astype(int)
        self.noisy.astype(int)
        
    
    def save_eventdf(self, path):
        self.eventdf.to_csv(path, index=False)
    
    def save_clusterdf(self, path):
        self.clusterdf.to_csv(path, index=False)
        
        
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

    
    
    # takes dataframe of a single cluster and finds line of best fit
    def fitline(self, cluster, pop_num=50, gen_num=200, verbose=False):
        
        print("using old...")
        
        hit_chs = cluster['Channel'].values
        hit_chs = hit_chs[np.isin(hit_chs, self.noisy, invert=True)]

        non_sat_chs = cluster[cluster['IsSaturated'] == False]['Channel'].values
        non_sat_chs = non_sat_chs[np.isin(non_sat_chs, self.noisy, invert=True)]
        
        if len(hit_chs) < 3:
            return self.basicfit(cluster)
        
        miss_chs = np.array([ch for ch in range(1,989) if ch not in hit_chs])
        miss_chs[np.isin(miss_chs, self.dead, invert=True)]
        
        PRELIM_POP = 100
        PRELIM_GENS = 100
        
        #prelim_ref_dirs = get_reference_directions("energy", 2, PRELIM_POP) #, seed=1)
            
        prelim_algorithm = NSGA2( #UNSGA3
            pop_size=PRELIM_POP,
            #ref_dirs=prelim_ref_dirs,
            n_offsprings=PRELIM_POP,
            sampling=get_sampling("real_random"),#self.initial_sample(hit_chs,PRELIM_POP), 
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
        
        #ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=200)
        ref_dirs = get_reference_directions("energy", 3, pop_num)
        
        algorithm = NSGA3(
            pop_size=pop_num,
            ref_dirs=ref_dirs,
            #n_offsprings=pop_num,
            sampling=prelim_result.X, #get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=1.0, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        
        termination = get_termination("n_gen", gen_num)
        
        problem = MyProblem(hit_chs, miss_chs, non_sat_chs, cluster['SelectedEnergy'].values, num_obj=3)
        #problem = MyProblem()
        
        res = minimize(problem,
                       algorithm,
                       termination,
                       #seed=1
                       #pf=problem.pareto_front(use_cache=False),
                       #save_history=True)
                       verbose=verbose
                      )
        if verbose:
            print(res.X)
            print(res.F)
            
#         f = 0 * res.F[:,0] + 0 * res.F[:,1] + res.F[:,2] # possibly add weights here?
#         sorted_f = sorted(f)
        
        bestline = []
        #bestscore = np.inf
        
        best_missed = np.inf
        best_extra = np.inf
        best_linear = np.inf
        
#         bestline = res.X[np.where(f==sorted_f[0])[0][0]]
                
        
        
       #if sorted_f[0] != 1.0:
       #     bestline = res.X[np.where(f==sorted_f[0])[0][0]] != 1.0
       # 
        #else:
        
        if not hasattr(res, 'X') or res.X is None:
            print("Many-Objective Failure, defaulting to Multi-Objective...")
            
            res = prelim_result
            res.F = np.insert(prelim_result.F, 2, np.inf, axis=1)
        
        #find best line out of top ten
        for i in range(len(res.X)):
        #for betterline in res.X:

#             index = np.where(f==sorted_f[i])[0][0]

            betterline = pts_to_line(res.X[i])
            hit_channels = channelcollisions(betterline, self.coords)[0]
            missed, extra = self.errorchannels(cluster, hit_channels)
            linear = res.F[i][2]

            if verbose and len(missed) == 0 and len(extra) == 0:
                print(betterline, res.F[i])


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


            #if best_missed + best_extra == 0:
            #    bestline[3:] = bestline[3:] / np.linalg.norm(bestline[3:])
            #    return bestline


                
            #print((best_missed, best_extra))

            #if len(extra) + len(missed) < bestscore:
            #    bestline = betterline
            #    bestscore = len(extra) + len(missed)
            #    print(bestscore)
            
        '''
                
        #normalize direction vector
        bestline[3:] = bestline[3:] / np.linalg.norm(bestline[3:])
        
        # at end of analysis, check if we are any better off than how we started
        basicline = self.basicfit(cluster)
        hit_channels = self.channelcollisions(basicline)[0]
        basic_missed, basic_extra = self.errorchannels(cluster, hit_channels)
        
        # also maybe consider flagging the fact that we used basicfit line?
        #if len(basic_extra) + len(basic_missed) < bestscore:
        #        print("NSGA2 failure...defaulting to LSR")
        #        bestscore = len(basic_extra) + len(basic_missed)
        
        
        if len(basic_missed) < best_missed:
            print("NSGA2 failure...defaulting to LSR")
            best_missed = len(basic_missed)
            best_extra = len(basic_extra)
            bestline = basicline

        elif len(basic_missed) == best_missed and len(basic_extra) < best_extra:
            print("NSGA2 failure...defaulting to LSR")
            best_extra = len(basic_extra)
            bestline = basicline
        
        '''
    
        return np.round(bestline, decimals=6)
    
        
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
            
        
#         non_sat_chs = np.array(non_sat_chs)

#         data = []
#         for i in range(len(non_sat_chs)):
#             if non_sat_chs[i] in hit_channels:
#                 data.append((track_distances[np.where(hit_channels == non_sat_chs[i])[0][0]], energies[i]))
#             #else:
#             #    data = []
#             #    break   

#         data = np.array(data)

#         x = data[:,0][:,np.newaxis]            
#         y = data[:,1]

#         non_sat_dEdxs = []
#         for xi, yi in zip(x,y):
#             dEdx = yi / xi
#             non_sat_dEdxs.append(dEdx)
        
#         all_dEdxs = []
#         for i, row in cluster.iterrows():
#             if not row['IsSaturated'] and row['Channel'] not in self.noisy:
#                 all_dEdxs.append(non_sat_dEdxs.pop(0))
#             else:
#                 all_dEdxs.append(np.nan)





                
#         if len(data) == 0:
#             m, r2 = (0.0, 0.0)
#         elif len(data) == 1:
#             m, r2 = (data[:,1][0] / data[:,0][0], 1.0)
        
#         else:
#             x = data[:,0][:,np.newaxis]            
#             y = data[:,1]

#             slope, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
#             m = slope[0]
#             r2 = r2_score(y, data[:,0] * m)
            
        
#         if show_graph:
#             plt.scatter(data[:,0], data[:,1], color='b')
#             #plt.plot(np.linspace(0,70), np.linspace(0,70)*m, label='r2='+str(r2))
            
#             plt.ylabel("Selected Energy (keV)")
#             plt.xlabel("Path length (mm)")
#             plt.legend()
            
#             plt.show()
        
        return dEdxs
        #return (m, r2)

    #return stats.linregress(data[:,0], data[:,1])

    
    
    
    
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
       
    
    def make_clusterdf(self, pop_num=None, gen_num=None, basicfit = False, verbose=False):
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
        #dEdxs = []
        #dEdx_errs = []
        
        #add columns to eventdf
        
        self.eventdf['Hit'] = False
        self.eventdf['PathLength'] = np.nan
        self.eventdf['dEdx'] = np.nan
        
        for c in clusters:

            cluster = self.eventdf[self.eventdf['Cluster'] == c]

            #event and channel info
            if len(cluster) == 1 and cluster['Channel'].values[0] == -1:
                eventspercluster.append(0)
                channelspercluster.append(0)
            else:
                eventspercluster.append(len(cluster))
                channelspercluster.append(len(cluster['Channel'].unique()))

            #get timespread
            clustertimes = cluster['MaxTime']
            starttimes.append(min(clustertimes))
            timespreads.append(max(clustertimes) - min(clustertimes))

            #get fitline
            if basicfit:
                fitline = self.basicfit(cluster)
            elif pop_num and gen_num:
                fitline = self.fitline(cluster, pop_num, gen_num, verbose)
            elif not pop_num and not gen_num:
                fitline = self.fitline(cluster)
            else:
                print("Error: fitline unspecficied")
                sys.exit()
                
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
            #dEdx, err = self.dEdx(cluster, fitline)
            #dEdxs.append(dEdx)
            #dEdx_errs.append(err)
        
        #zeniths_degrees = [theta*360/(2*math.pi) for theta in zeniths]
        #cos_theta = [math.cos(theta) for theta in zeniths]

        d = {'Cluster' : clusters, 'Events' : eventspercluster, 'Channels' : channelspercluster, \
            'StartTime': starttimes, 'TimeSpread' : timespreads, 'NRMSE' : NRMSE, \
             'Zenith' : zeniths,'Azimuth': azimuths, 'ExtraCh': extrachannels, 'MissingCh' : missingchannels, \
             'Fitline' : fitlines}
             #'dEdx': dEdxs, 'dEdx_err': dEdx_errs, 'Fitline' : fitlines}

        #return newdf
        self.clusterdf = pd.DataFrame(data=d)
        
        return self
        
        
    def filter_fit(self, NRMSE, channels):
        self.clusterdf = self.clusterdf[self.clusterdf['NRMSE'] < NRMSE]
        self.clusterdf = self.clusterdf[self.clusterdf['Channels'] >= channels]
        
        if hasattr(self, 'eventdf'):
            self.eventdf = self.eventdf[self.eventdf['Cluster'].isin(self.clusterdf['Cluster'].values)]
        
        return self


    def get_clusterdf(self):
        return copy.deepcopy(self.clusterdf)
    
    
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
    
    
    def show_channel(self, channel_list, x1=15, x2=45):
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_proj_type('ortho')
        
        if isinstance(channel_list, (int, np.int64)):
            channel_list = [channel_list]
            
        coords = np.array([self.coords[ch] for ch in channel_list]).T
        ax.scatter3D(*coords)
        
        plt.xlim([-350,350])
        plt.ylim([-350,350])
        ax.set_zlim([-350,350])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(x1, x2)
        
        plt.show()
        
    
        
    def show_cluster(self, cluster_list, x1=15, x2=45):
        
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_proj_type('ortho')
        
        #if given int, make list
        if isinstance(cluster_list, (int, np.int64)):
            cluster_list = [cluster_list]
            
        for c in cluster_list:
            cluster = self.eventdf[self.eventdf['Cluster'] == c]

            coords = self.clustercoords(cluster)
            
            ax.scatter3D(*coords)

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

        ax.view_init(x1, x2)
        
        plt.show()
        
    def show_simulation(self, cluster_num, x1=15, x2=45):
        
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

        ax.view_init(x1, x2)

        plt.show()
        