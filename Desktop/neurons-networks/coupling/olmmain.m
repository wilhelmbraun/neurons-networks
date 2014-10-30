close all
%ic1=[-75.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.1];
dt=0.05;

%%% determine spike times on post-synaptic neuron
I150=-4.70;
%-4.70; % Current that evokes 150ms period
spikeThreshold=10.;

tf=500;
% G = 0;
% Iapp=@(t) I150;
% 
% ic=[-71.6155713624969,0.0201933303996961,0.844749455804224,0.0831039963077916,0.0339725442676448,0.214086668247333,0.0163650253912917,0.402210373337459];
% tspan=0:dt:5000;
% tic;
% [T,Y]=ode15s(@(t,y) HH(t,y,G,Iapp) , tspan, ic);
% toc;
% 
% V=Y(:,1);
% spikeTimes1=findSpikeTimes(V,dt,spikeThreshold);
% Tau=mean(diff(spikeTimes1));

figure(1)
subplot(3,1,1)
plot(T,Y(:,1),'k-')
hold on
% 
 %Tau=150.324461538462; %Value found from above spike train with many spikes
%return;
%%% determine phase response
%ic1=[-71.4502511076917,0.0209131239631594,0.836088751403759,0.0929572650458564,0.0304469014757546,0.515386761203573,0.226104787390827,0.];
%ic1=[-67.0519499310160,0.0361697241475829,0.727344983888989,0.104286490800503,0.0393202822722456,0.367994816539880,0.159550881948505,2.77555756156212e-13];
%ic1=[-56.5185719801193,0.121341547906334,0.367988704271232,0.150799277018200,0.0716735238306508,0.120770945648591,0.0632980958371113,5.35101137793166e-11];
ic1=[-56.5189760324990,0.121308490680905,0.368074239602490,0.150791746974933,0.0716654043134373,0.120924764929777,0.0633153985638902,5.33406655157451e-11];
ic2=[-71.6155713624969,0.0201933303996961,0.844749455804224,0.0831039963077916,0.0339725442676448,0.214086668247333,0.0163650253912917,0.402210373337459];
%ic2=[-75.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.1];

[~,dim]=size(ic1);
ic=horzcat(ic1,ic2);
tspan=0:dt:tf;

% coupling matrix
g21=.5; % 2 receives from 1
G = [0 0; g21 0]; % coupling matrix
N = size(G,1);

% applied current parameters
NSp=1;
t0=spikeTimes1(NSp); % set t0 to the time of some arbitrary spike
thetaAmp=-15.;
preSynSpikeAmp=10.;
phaseStep=1;

% iterate through discrete phase values (1ms)
P=1:phaseStep:floor(Tau);
f=zeros(1,length(P));
delta=zeros(1,length(P));
delta_extra=0;
f_extra=0;
extra_count=0;
for p=1:length(P);
    pulseOn=t0+P(p)
    pulseOff=pulseOn+7.;

    % applied current vector
    Iapp=@(t)[preSynSpikeAmp*heaviside(t-pulseOn)*heaviside(pulseOff-t)+thetaAmp,I150];
    %Iapp=@(t)[preSynSpikeAmp*(1+tanh(20*(t-pulseOn)))*(1-tanh(20*(t-pulseOff)))/2+thetaAmp,I150];
    %Iapp=@(t)[preSynSpikeAmp*heaviside(t-pulseOn)+thetaAmp,I150];
    %Iapp=@(t)[-8,I150];
    
    tic;
    [T,Y]=ode15s(@(t,y) HH(t,y,G,Iapp), tspan, ic);
    toc;

    postSpikeTimes=findSpikeTimes(Y(:,1+dim),dt,spikeThreshold);
    t1=findSpikeTimes(Y(:,1),dt,spikeThreshold);
    
    if fix((t1-t0)./Tau)>0
        extra_count=extra_count+1;
        delta_extra(extra_count)=rem(t1-t0,Tau); % 1 spike => 1 entry
        t2=postSpikeTimes(NSp+2);
        f_extra(extra_count)=t2-postSpikeTimes(NSp+1)-Tau;
    else
        delta(p)=t1-t0;
        t2=postSpikeTimes(NSp+1);
        f(p)=t2-t0-Tau;
    end
end

delta=horzcat(delta_extra,delta);
f=horzcat(f_extra,f);

% figure 1 plotting 
subplot(3,1,2);
plot(T,Y(:,1))
ylabel('V')

subplot(3,1,3)
plot(T,Y(:,1+dim))
xlabel('t')
hold on
plot(spikeTimes1,35*ones(size(spikeTimes1,2)),'+')

figure()
clf;
plot(delta,f,'k-')