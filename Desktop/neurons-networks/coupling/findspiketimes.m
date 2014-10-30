function t = findSpikeTimes(V,dt,thresh)
    tCrossThreshold=find(V>thresh); %find points that lie above -10mV
    spikeTimes=zeros(1,size(V,1));
    spikeStart=0;
    k=0;
    j=1;
    Tau=1; %Approximate number of timesteps between sets
    s=size(tCrossThreshold,1);
    for i=1:s
        if (k==0)
            spikeStart=tCrossThreshold(i);
            k=k+1;
        elseif ((i>1 && abs(tCrossThreshold(i)-tCrossThreshold(i-1)) > Tau) || i==s) 
            spikeTimes(j)=dt*(tCrossThreshold(i-1)+spikeStart)/2.;
            spikeStart=tCrossThreshold(i);
            j=j+1;
            k=0;
        else
            k=k+1;
        end
    end
    t=spikeTimes(1:j-1);
end