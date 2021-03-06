import numpy as np
from scipy import stats
from pymoo.util.misc import stack
from pymoo.model.problem import Problem

from sklearn.metrics import r2_score

from Helper import *

#pd.options.display.precision = 2


CUBELENGTH = 50
INSIDE_BOUND = CUBELENGTH/2*np.sqrt(3)

class MyProblem(Problem):

    def __init__(self, hit_chs, miss_chs, non_sat_chs, energies, num_obj):
        
        self.pwd = "."
        #self.pwd = str(Path().absolute())
        #self.pwd = "/nfs/cuore1/scratch/yocum"
        self.coords = load_coords(self.pwd)
        
        self.hit_chs = hit_chs
        self.miss_chs = miss_chs
        self.non_sat_chs = non_sat_chs
        
        self.energies = energies

        self.num_obj = num_obj
        
        self.hit_pts = np.array([self.coords[pt] for pt in hit_chs]).T
        self.miss_pts = np.array([self.coords[pt] for pt in miss_chs]).T
        
        boundary = self.boundary_coords()
        
        self.dEdx_pdf = make_pdf(self.pwd + '/data/pdf/dEdx/bins_dEdx_muon.csv', self.pwd + '/data/pdf/dEdx/values_dEdx_muon.csv', domain_range=[0,190])
        
        xl = np.copy(boundary) - 26
        xu = np.copy(boundary) + 26
        
        super().__init__(n_var=6,
                         n_obj=self.num_obj,
                         n_constr= 0 if self.num_obj == 2 else 1,
                         xl=xl,
                         xu=xu)
        

    def _evaluate(self, x, out, *args, **kwargs):
        
        #print(x)
        
        if self.num_obj == 2:
        
            f1 = self.hitcost(x)
            f2 = self.misscost(x)
            
            out["F"] = np.column_stack([f1, f2])
            
        
        elif self.num_obj == 3:
            
            f1 = self.hitcost(x)
            f2 = self.misscost(x)
            f3 = self.fitcost(x)

            g = self.hitcost(x) + self.misscost(x) - 1/20

            out["F"] = np.column_stack([f1, f2, f3])
            out["G"] = np.column_stack([g])
        
        
        
    def boundary_coords(self):
        
        p = self.hit_pts.T.mean(axis=0)
        uu, dd, vv = np.linalg.svd(self.hit_pts.T - p)  
        v = vv[0] / np.linalg.norm(vv[0])
        line = np.append(p, v)
        linepts = line_to_pts(line).flatten()
        
        hit_distances = self.ptsfromline(self.hit_pts, linepts.flatten()) 
        indices = [pt for dist,pt in sorted(zip(hit_distances, np.arange(len(hit_distances))))]
        best_pts = np.append(self.hit_pts.T[indices[0]], self.hit_pts.T[indices[1]])
            
        return best_pts
    
        
    #all_linepts = np.tile(linepts.T, (3,1,1))
    def ptsfromline(self, pts, linepts):
        
        a = linepts[np.newaxis].T[:3]
        b = linepts[np.newaxis].T[3:]

        d = np.linalg.norm(np.cross(pts - a, pts-b, axis=0), axis=0) / np.linalg.norm(b-a, axis=0)
        
        return d
    
#     def hitcost (self, x): 
    
#         costs = []
        
#         cubeLength = 50
#         inside = cubeLength/2*np.sqrt(3)

#         bound = (cubeLength + inside) / 2

#         for line in x:

#             #p = line[:3]
#             #v = line[3:]

#             #linepts = v * np.mgrid[-.5:.5:2j][:, np.newaxis] + p
                         
#             linepts = line

#             hitlist = self.ptsfromline(self.hit_pts, linepts)

#             #return sum([d**2 for d in hitlist]) + sum([inside**4/d**2 for d in misslist])

#             linecost =  sum([1/(1 + np.exp(-.2*(d-inside))) for d in hitlist]) 
#             linecost = linecost / len(hitlist)
#             costs.append(linecost)
            
#         return np.array(costs)

    def hitcost (self, x): 
    
        costs = []
        
        #BOUND = (cubeLength + INSIDE_BOUND) / 2

        for linepts in x:
            hitlist = self.ptsfromline(self.hit_pts, linepts)

            #return sum([d**2 for d in hitlist]) + sum([inside**4/d**2 for d in misslist])

            #linecost =  sum([1/(1 + np.exp(-.2*(d-inside))) for d in hitlist]) 
            linecost = np.sum(1 / (1 + np.exp(-.2*(hitlist - INSIDE_BOUND))))
            linecost = linecost / len(hitlist)
            
            costs.append(linecost)
            
        return np.array(costs)



#     def misscost (self, x): 

#         costs = []
                                   
#         cubeLength = 50
#         inside = cubeLength/2*np.sqrt(3)

#         bound = (cubeLength + inside) / 2

#         for line in x:
#             #p = line[:3]
#             #v = line[3:]

#             #linepts = v * np.mgrid[-.5:.5:2j][:, np.newaxis] + p
                         
#             linepts = line

#             misslist = self.ptsfromline(self.miss_pts, linepts)

#             #return sum([d**2 for d in hitlist]) + sum([inside**4/d**2 for d in misslist])

#             linecost =  sum([1/(1 + np.exp(.2*(d-25))) for d in misslist])
#             linecost = linecost / len(misslist)
#             costs.append(linecost)
            
#         return np.array(costs)
    
    def misscost (self, x): 

        costs = []

        for linepts in x: 

            misslist = self.ptsfromline(self.miss_pts, linepts)
            #return sum([d**2 for d in hitlist]) + sum([inside**4/d**2 for d in misslist])

            linecost = np.sum(1 / (1 + np.exp(.2*(misslist - 25))))
            linecost = linecost / len(misslist)
            
            costs.append(linecost)
            
        return np.array(costs)
    
    def fitcost (self, x):
        
        costs = []
        
        #x = np.array([[x1, y1, z1, x2, y2, z2)]
        #                [})
        
        for linepts in x:
            
            line = pts_to_line(linepts)
            hit_channels, _, track_distances = channelcollisions(line, self.coords)
            
            data = []
            for i in range(len(self.hit_chs)):
                if self.hit_chs[i] in hit_channels and self.hit_chs[i] in self.non_sat_chs:
                    data.append((track_distances[np.where(hit_channels == self.hit_chs[i])[0][0]], self.energies[i]))
                    
            data = np.array(data).transpose()
            
            if len(data) == 0:
                costs.append(0)
                continue
            else:    

                dEdx = data[1] / data[0]
                p_densities = self.dEdx_pdf(dEdx/100)
                log_densities = np.log(p_densities)
                
                #print(dEdx)
                cost = - log_densities.sum()
                costs.append(cost)
                #print(dEdx/100, p_densities, cost)

        return np.array(costs)
    
    
#     def lincost (self, x):
        
#         costs = []
        
#         for linepts in x:
            
#             line = pts_to_line(linepts)
#             hit_channels, _, track_distances = channelcollisions(line, self.coords)
            
            
            
#             data = []
#             for i in range(len(self.hit_chs)):
#                 if self.hit_chs[i] in hit_channels and self.hit_chs[i] in self.non_sat_chs:
#                     data.append((track_distances[np.where(hit_channels == self.hit_chs[i])[0][0]], self.energies[i]))
#                 #else:
#                 #    data = []
#                 #    break   
                    
#             data = np.array(data)
#             modifier = len(self.non_sat_chs) - len(data) + 1
            
#             '''
#             print(line)
#             print(hit_channels)
#             print(self.hit_chs)
#             print(self.non_sat_chs)
#             print(self.energies)
#             print(data)
#             '''
            
#             if len(data) in [0,1]:
#                 costs.append(modifier)
#                 continue
#             else:    
#                 x = data[:,0][:,np.newaxis]            
#                 y = data[:,1]

#                 slope, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
#                 m = slope[0]
#                 r2 = r2_score(y, data[:,0] * m)
                
#                 costs.append(-r2 + modifier)
                
            
#             #slope, intercept, r_value, p_value, std_err = stats.linregress(data[:,0], data[:,1])
                        
#             #if r2 == 0:
#             #    costs.append(modifier)
#             #else:
#             #    costs.append((1 / r2) * modifier)
            
#             #costs.append(-r2 + modifier)
            
            
#         return np.array(costs)