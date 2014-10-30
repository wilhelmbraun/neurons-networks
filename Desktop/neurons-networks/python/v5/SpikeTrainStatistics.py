import numpy as np

#Determination of spike times from numerically obtained spike trains
#INPUT: V: spike train, dt: integration time step, thresh: threshold value obtained by visual inspection of the spike train


def findSpikeTimes(V,dt,thresh):
    
    #find points above a certain threshold
    tCrossThreshold=np.where(V>thresh)
    
    #print tCrossThreshold[0][:]
    
    spikeTimes=[]
    
    spikeStart=0
    
    #approximate number of time steps between spikes
    Tau=1
    spikeStart=tCrossThreshold[0][0]
    for i in range(0, np.size(tCrossThreshold)):
        
        
        if ( i >=1 and (np.abs(tCrossThreshold[0][i]-tCrossThreshold[0][i-1]) > Tau) or i==np.size(tCrossThreshold) -1):
        
            if i==np.size(tCrossThreshold) -1:
                spikeTimes = np.append(spikeTimes,dt*((tCrossThreshold[0][i]+spikeStart)/2.))

            else:
                spikeTimes = np.append(spikeTimes,dt*((tCrossThreshold[0][i-1]+spikeStart)/2.))
            

            
            spikeStart=tCrossThreshold[0][i]
            
                
    return spikeTimes
