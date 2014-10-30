from networks import OLMFSNetwork
from matplotlib import pyplot as plt
import numpy as np

# Index 0,1= OLM Neurons
# Index 2 = Interneuron
"""
setting parameters. No noise.
"""
dt = 0.025
tf=1000.
T=int(np.floor(tf/dt))
time = np.linspace(0, tf, T)

# synaptic reversal potential matrix
v_gaba=-80.
V_rev=v_gaba*np.ones(3)

# synaptic conductance matrix
G_II=0.
G_IO=0.2 #0.2
G_OI=0.08 #0.
G_OO=0.2 #0.01 #0.01
G=np.asfarray([[0., G_OO, G_IO],[G_OO, 0., G_IO],[G_OI,G_OI,G_II]])

# applied current vector
I_O=-1.8
I_I=0.5 #0.154 #0.52 #0.154
I = np.asfarray([I_O,I_O,I_I])

#starting conditions for  [v, m, h, n, a, b, r, s]
ic1=[ -66.1133769,5.29112479e-03 ,9.80308296e-01 ,1.22324042e-01, 1.28808956e-02 ,4.57163233e-02 ,2.99566654e-02 ,5.68180524e-01]
ic2=[ -50.1133769,5.29112479e-03 ,9.80308296e-01 ,1.22324042e-01, 1.28808956e-02 ,4.57163233e-02 ,2.99566654e-02 ,5.68180524e-01]
#ic2=[-75.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.4]
#ic3=[-50.61,0.0122,0.9152,0.07561,0.0229]
ic3=[-65.4451181,2.00667975e-02 ,9.95604331e-01 ,4.36785808e-02,6.85519809e-02]
ic=[ic1,ic2,ic3]
network = OLMFSNetwork(ic, G, I, V_rev)

n_neuron=network.size

v_data = np.zeros([n_neuron,T])  
syn_data = np.zeros([n_neuron,T])     
gh_data = np.zeros([2,T])
state_data1=np.zeros([len(network.neurons[0].state), T])
state_data2=np.zeros([len(network.neurons[1].state), T])
state_data3=np.zeros([len(network.neurons[2].state), T])

for t in xrange(T):
    network.timestep(dt)
    v_data[:,t] = network.v_position()
    state_data1[:,t] = network.neurons[0].state
    state_data2[:,t] = network.neurons[1].state
    state_data3[:,t] = network.neurons[2].state
    #gh_data[:,t] = network.OLMih()
    #syn_data[:,t] = network.synaptic_position()
    
print state_data3[:,-1]

plt.figure(1)
ax = plt.subplot(311,xlim=(0,T * dt),  ylim=(-100,60))    
ax.plot(time, v_data[0],lw=1,label='OLM neuron 0',color='b')
#ax = plt.axes(xlim=(0,T * dt),  ylim=(-100,60))

#ax.set_title('$G_{OO}=0.5$')
 
#lines = [ax.plot(time, v_data[i], lw=1, label = 'OLM neuron %s'%(i) ) for i in xrange(2)]
plt.legend(loc = 'best')

ax2 = plt.subplot(312, xlim=(0,T * dt),  ylim=(-100,60))
ax2.plot(time, v_data[1],lw=1,label='OLM neuron 1',color='g')
#ax2.plot(time, v_data[2],lw=1,label='FS neuron 1',color='r')
ax2.set_ylabel('$V$ (mV)')
plt.legend(loc = 'best')

ax3 = plt.subplot(313, xlim=(0,T*dt), ylim=(-100,60))
#ax3 = plt.subplot(313, xlim=(0,T*dt), ylim=(-10,10))
ax3.plot(time, v_data[2],lw=1,label='FS neuron 1',color='r')
#l = [ax3.plot(time, gh_data[i],lw=1,label='OLM $I_{h,%s}$' % (i)) for i in xrange(2)]
ax3.set_xlabel('$t$ (ms)')
plt.legend(loc='best')

#ax = plt.axes(xlim=(0,T * dt), ylim=(-1,2))

#plt.figure(2)
#lines2 = [plt.plot(time, syn_data[i], lw=2, label = 'synapse %s'%(i) ) for i in xrange(neurons.size)]

#plt.legend(loc = 'best')
plt.show()