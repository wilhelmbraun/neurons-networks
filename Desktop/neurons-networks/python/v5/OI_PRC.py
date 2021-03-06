from networks import OLMFSNetwork
from matplotlib import pyplot as plt
import numpy as np

"""
setting parameters. No noise.
"""
dt = 0.025
tf=400.
T=int(np.floor(tf/dt))
time = np.linspace(0, tf, T)

# synaptic reversal potential vector
v_gaba=-80.
#V_rev=np.asfarray([[v_gaba,0.],[0.,v_gaba]])
V_rev=np.asfarray([v_gaba,v_gaba])

# synaptic conductance matrix
g21=0.0
g12=0.
G=np.asfarray([[0.,g12],[g21,0.]])

# applied current vector
I1=-2.25
I2=0.
I = np.asfarray([I1,I2])

#starting conditions for  [v, m, h, n, a, b, r, s]
ic1=[ -5.73437892e+01,   1.67052022e-02  , 9.70111287e-01 ,  8.27456026e-02,
   4.84096232e-02,   7.02289575e-02 ,  5.19345088e-02  , 1.06714013e-01]
ic2=[ -6.65910934e+001  , 1.60427822e-002 ,  9.95496069e-001  , 4.02751240e-002,
   2.09612817e-145]
#ic1=[-75.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.4]
#ic2=[-65.4451181,2.00667975e-02 ,9.95604331e-01 ,4.36785808e-02,6.85519809e-02]
ic=[ic1,ic2]
network = OLMFSNetwork(ic, G, I, V_rev,1,1)

n_neuron=network.size

v_data = np.zeros([n_neuron,T])  
syn_data = np.zeros([n_neuron,T]) 

for t in xrange(T):
    network.timestep(dt)
    v_data[:,t] = network.v_position()
    #syn_data[:,t] = network.synaptic_position()
    
#print v_data
print network.neurons[0].state
print network.neurons[1].state
    
ax = plt.axes(xlim=(0,T * dt),  ylim=(-100,60))

plt.figure(1)  
lines = [ax.plot(time, v_data[i], lw=2, label = 'neuron %s'%(i) ) for i in xrange(n_neuron)]

plt.legend(loc = 'best')
#ax = plt.axes(xlim=(0,T * dt), ylim=(-1,2))

#plt.figure(2)
#lines2 = [plt.plot(time, syn_data[i], lw=2, label = 'synapse %s'%(i) ) for i in xrange(neurons.size)]

#plt.legend(loc = 'best')
plt.show()