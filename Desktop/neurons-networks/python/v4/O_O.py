from networks import OLMNetwork
from matplotlib import pyplot as plt
import numpy as np

"""
setting parameters. No noise.
"""
dt = 0.05
tf=1000.
T=int(np.floor(tf/dt))
time = np.linspace(0, tf, T)

# synaptic reversal potential vector
v_gaba=-80.
#V_rev=np.asfarray([[v_gaba,0.],[0.,v_gaba]])
V_rev=np.asfarray([v_gaba,v_gaba])

# synaptic conductance matrix
g21=0.1
g12=0.1
G=np.asfarray([[0.,g12],[g21,0.]])

# applied current vector
I1=-4.7
I2=-4.7
I = np.asfarray([-4.7,-4.7])

#starting conditions for  [v, m, h, n, a, b, r, s]
ic1=[-75.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.4]
ic2=[-73.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.4]
ic=[ic1,ic2]
network = OLMNetwork(ic, G, I, V_rev)

n_neuron=network.size

v_data = np.zeros([n_neuron,T])  
syn_data = np.zeros([n_neuron,T])     

for t in xrange(T):
    network.timestep(dt)
    v_data[:,t] = network.v_position()
    #syn_data[:,t] = network.synaptic_position()
    
print v_data
    
ax = plt.axes(xlim=(0,T * dt),  ylim=(-100,60))

plt.figure(1)  
lines = [ax.plot(time, v_data[i], lw=2, label = 'neuron %s'%(i) ) for i in xrange(n_neuron)]

plt.legend(loc = 'best')
#ax = plt.axes(xlim=(0,T * dt), ylim=(-1,2))

#plt.figure(2)
#lines2 = [plt.plot(time, syn_data[i], lw=2, label = 'synapse %s'%(i) ) for i in xrange(neurons.size)]

#plt.legend(loc = 'best')
plt.show()