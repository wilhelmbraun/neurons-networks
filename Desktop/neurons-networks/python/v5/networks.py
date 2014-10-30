"""
date: 12.07.2014

Simulation of single determinstic and stochastic Hodgkin-Huxley type neurons
"""
import numpy as np
from numpy import random as rnd

from neurons import HoHuxNeuron,OLMneuron,O_Rotstein,I_Rotstein

class Network:
    def __init__(self, init_state = [], G=None, I=None, V_rev=None):
        self.size = len(init_state)
        self.neurons = []
        self.time_elapsed = 0
	
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
	#V=np.eye(N)*self.V_rev-np.eye(N)*v
	#interaction1=self.I+np.dot(V,np.dot(self.G,s))
	
	V=self.V_rev-v
	interaction=self.I + V*np.dot(self.G,s)
	
	#print interaction==interaction1
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
    
class OLMNetwork(Network):
    """
    Class for a network of OLM neurons
    The init_state is [v0, h0, n0, s0, coupling, N] where
    
    initial conditions are as above
    coupling is a coupling matrix (identity for synaptic coupling, see below)
    N is the number of neurons
    """
    def __init__(self,init_state=[],G=None,I=None,V_rev=None):
	Network.__init__(self,init_state, G, I, V_rev)
	
	for i in xrange(self.size):
            self.neurons.append(OLMneuron(init_state[i]))
	
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

class OLMFSNetwork(Network):
    """
    Class for coupling an OLM neuron to an FS neuron
    The init_state is [v0, h0, n0, s0, coupling, N] where
    
    initial conditions are as above
    coupling is a coupling matrix (identity for synaptic coupling, see below)
    N is the number of neurons
    """
    def __init__(self,init_state=[],G=None,I=None,V_rev=None,NO=2,NI=1):
	Network.__init__(self,init_state, G, I, V_rev)
	self.NO=NO
	self.NI=NI
	for i in xrange(NO):
            self.neurons.append(O_Rotstein(init_state[i],sigma=0))
	for i in xrange(NO,NO+NI):
	    self.neurons.append(I_Rotstein(init_state[i]))
	    
    def v_position(self):
        state = []
        for i in xrange(self.size):
            state.append(self.neurons[i].state[0])
        
        return np.asfarray(state)      
    
    def synaptic_position(self):
	state = []
	for i in xrange(self.size):
	    state.append(self.neurons[i].state[-1])
	
	return np.asfarray(state)     
    
    def OLMih(self):
	state = []
	for i in xrange(self.NO):
	    state.append(self.neurons[i].getIh())
	
	return np.asfarray(state)
     
    