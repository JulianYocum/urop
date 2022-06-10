from DataAnalysis import DataAnalysis
from MonteCarlo import MonteCarlo
import numpy as np
#import copy

#added in because of editors comments
def getTestDataframes(df_mc, saturation_cap):
    df_perfect = df_mc.copy()
    df_perfect.drop(['dEdx', 'PathLength', 'Hit'], axis=1, inplace=True)

    df_ideal = df_perfect.copy()    
    energies = np.array(df_mc['SelectedEnergy'])
    energies_with_noise = energies + \
        (energies < saturation_cap) * np.random.normal(0,10 / 2.355,len(energies)) + \
        (energies >= saturation_cap) * np.random.normal(0, energies * .10 / 2.355, len(energies))
    df_ideal['SelectedEnergy'] = energies_with_noise
    
    df_cuore = df_ideal.copy()
    df_cuore['IsSaturated'] = df_cuore['SelectedEnergy'] >= saturation_cap
    
    return (df_perfect, df_ideal, df_cuore)

mc = MonteCarlo(num_tracks=2000, track_type="sasso", particle_type="muon")
mc.filter_fit(100, 3)
mc.save_eventdf("/nfs/cuore1/scratch/yocum/data/MC_events.csv")
mc.save_clusterdf("/nfs/cuore1/scratch/yocum/data/MC_clusters.csv")

df_perfect, df_ideal, df_cuore = getTestDataframes(mc.get_eventdf(), 20000)

mc.set_eventdf(df_perfect)
mc.save_eventdf("/nfs/cuore1/scratch/yocum/data/perfect_raw_events.csv")
mc.set_eventdf(df_ideal)
mc.save_eventdf("/nfs/cuore1/scratch/yocum/data/ideal_raw_events.csv")
mc.set_eventdf(df_cuore)
mc.save_eventdf("/nfs/cuore1/scratch/yocum/data/cuore_raw_events.csv")


print("saved MC data")

#da = DataAnalysis()
#da.set_eventdf(mc.get_eventdf())
#da.filter_noisy()
#da.arrange_clusters(e_thresh=3, t_thresh=1.0)
#da.save_eventdf("/nfs/cuore1/scratch/yocum/data/raw_events.csv")

head = MonteCarlo(0, 'isotropic')
head.save_clusterdf("/nfs/cuore1/scratch/yocum/data/perfect_clusters.csv")
head.save_clusterdf("/nfs/cuore1/scratch/yocum/data/ideal_clusters.csv")
head.save_clusterdf("/nfs/cuore1/scratch/yocum/data/cuore_clusters.csv")
head.save_eventdf("/nfs/cuore1/scratch/yocum/data/perfect_events.csv")
head.save_eventdf("/nfs/cuore1/scratch/yocum/data/ideal_events.csv")
head.save_eventdf("/nfs/cuore1/scratch/yocum/data/cuore_events.csv")
print("opened empty cluster/event files")

#df = da.get_cluster(0)


#da1 = DataAnalysis()
#da1.set_eventdf(df)
#da1.make_clusterdf(basicfit=True)

#da2 = DataAnalysis()
#da2.set_eventdf(df)
#da2.make_clusterdf(pop_num=200, gen_num=2000)

#da1.save_clusterdf("/nfs/cuore1/scratch/yocum/data/Basic_clusterdf.csv")
#da2.save_clusterdf("/nfs/cuore1/scratch/yocum/data/clusterdf.csv")
