"""
date: 12.07.2014

Simulation of single determinstic and stochastic Hodgkin-Huxley type neurons
"""

import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt

from neurons import HoHuxNeuron,OLMneuron

class Network:
    def __init__(self, init_state = []):
        self.size = len(init_state)
        self.neurons = []
        self.time_elapsed = 0
        
class HoHuxNetwork:
    """
    Class for a network of Hodgkin-Huxley neurons
    The init_state is [v0, h0, n0, s0, coupling, N] where
    
    initial conditions are as above
    coupling is a coupling matrix (identity for synaptic coupling, see below)
    N is the number of neurons
    """
    
    def __init__(self,
                 init_state = []
                 ):
        self.size = init_state[5]
        self.neurons = []
        
        for i in xrange(self.size):
            self.neurons.append(HoHuxNeuron([init_state[0][i], init_state[1][i], init_state[2][i], init_state[3][i]]))
            
        self.coupling = np.asarray(init_state[4], dtype='float')
        self.time_elapsed = 0
        
    def timestep(self, dt, I):
        """time evolution of the network"""
        interaction = []
        
        v_rev = -80
        g = 0.0
        
        for i in xrange(self.size):
	  
	  #second neuron gets synaptic input from first neuron
	  if i == 1: 
	   interaction.append( g*self.neurons[i-1].state[3] *(v_rev - self.neurons[i].state[0]))    
	   #interaction.append(0)
	   
	  #no coupling for first neuron
	  else:
	    interaction.append(0)
        
        #include current in the coupling, just a convention
	interaction = np.dot(self.coupling, interaction) + I
        
        for i in xrange(self.size):
            self.neurons[i].timestep(dt, interaction[i])
        
    def position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state)
        
        return state

    def v_position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state[0])
        
        return state        
    
    def synaptic_position(self):
	state = []
	for i in xrange(self.size):
	    state.append(self.neurons[i].state[3])
	
	return state      
	

class OLMNetwork(Network):
    """
    Class for a network of OLM neurons
    The init_state is [v0, h0, n0, s0, coupling, N] where
    
    initial conditions are as above
    coupling is a coupling matrix (identity for synaptic coupling, see below)
    N is the number of neurons
    """
    def __init__(self,init_state=[],G=None,I=None):
	Network.__init__(self,init_state)
	
	for i in xrange(self.size):
            self.neurons.append(OLMneuron(init_state[i]))
        
	self.G=G
	self.I=I
	self.V_rev=V_rev
	
    def timestep(self, dt):
        """time evolution of the network"""
        N=self.size
	
	interaction = np.zeros(N, dtype = 'float')
	
	s=self.synaptic_position()
	v=self.v_position()
	
	#print np.eye(N)*v
	V=np.eye(N)*self.V_rev-np.eye(N)*v
	interaction=self.I+np.dot(V,np.dot(self.G,s))
   
        '''
	interaction1 = np.zeros(N, dtype = 'float')
        v_rev = -80.
	g21=0.1
	g12=g21
	
	interaction1[0] = I[0] + g12*self.neurons[1].state[7]*(v_rev - self.neurons[0].state[0])
        interaction1[1] = I[1] + g21*self.neurons[0].state[7] *(v_rev - self.neurons[1].state[0])
	#print interaction==interaction1#for i in xrange(self.size):
	'''
	
        for i in xrange(self.size):
	    self.neurons[i].buildRHS(interaction[i])
            self.neurons[i].timestep(dt)
        
    def position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state)
        
        return state

    def v_position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state[0])
        
        return np.asfarray(state)      
    
    def synaptic_position(self):
	state = []
	for i in xrange(self.size):
	    state.append(self.neurons[i].state[7])
	
	return np.asfarray(state)      

class OLMFSNetwork:
    """
    Class for coupling an OLM neuron to an FS neuron
    The init_state is [v0, h0, n0, s0, coupling, N] where
    
    initial conditions are as above
    coupling is a coupling matrix (identity for synaptic coupling, see below)
    N is the number of neurons
    """
    
    def __init__(self,
                 init_state = []
                 ):
        self.size = init_state[9]
        self.neurons = []
        for i in xrange(self.size):
	  self.neurons.append(OLMneuron([init_state[0][i], init_state[1][i], init_state[2][i], init_state[3][i],
	      init_state[4][i], init_state[5][i], init_state[6][i], init_state[7][i]]))
	      
	  self.neurons.append(HoHuxNeuron([init_state[0][i], init_state[2][i], init_state[3][i], init_state[7][i]]))
            
        self.coupling = np.asarray(init_state[8], dtype='float')
        self.time_elapsed = 0
        
    def timestep(self, dt, I):
        """time evolution of the network"""
        interaction = []
        
        v_rev = -80
        
        #for i in xrange(self.size):
	  #if i ==1:
	    #interaction.append( g*self.neurons[i-1].state[7] *(v_rev - self.neurons[i].state[0]))    
	  #else:
	    #interaction.append(0)
	

	interaction[0] = 0. + I
	interaction[1] =  g* self.neurons[0].state[7] *(v_rev - self.neurons[1].state[0]) + I

        
        #include current in the coupling, just a convention
	#interaction = np.dot(self.coupling, interaction) + I
        
        for i in xrange(self.size):
            self.neurons[i].timestep(dt, interaction[i])
        
    def position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state)
        
        return state

    def v_position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state[0])
        
        return state        
    
    def synaptic_position(self):
	state = []
	for i in xrange(self.size):
	    state.append(self.neurons[i].state[3])
	
	return state      
    
    
"""
setting parameters. No noise.
"""
dt = 0.05
tf=1000.
T=int(np.floor(tf/dt))
time = np.linspace(0, tf, T)

# synaptic reversal potential matrix
v_gaba=-80.
V_rev=np.asfarray([[v_gaba,0.],[0.,v_gaba]])

# synaptic conductance matrix
g21=0.1
g12=0.1
G=np.asfarray([[0.,g12],[g21,0.]])

# applied current vector
I1=-4.7
I2=-4.7
I = np.asfarray([-4.7,-4.7])

#starting conditions for v, h, n, s, coupling, number of neurons
#neurons = HoHuxNetwork([[-100,-100], [0.1,0.5], [0.1, 0.5], [0.4, 0.4], np.eye(2), 2])

#starting conditions for  [v, m, h, n, a, b, r, s]
ic1=[-75.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.4]
ic2=[-73.61,0.0122,0.9152,0.07561,0.0229,0.2843,0.06123,0.4]
ic=[ic1,ic2]
network = OLMNetwork(ic, G, I)

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