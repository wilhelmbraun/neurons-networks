import numpy as np

class Neuron:
    def __init__(self,init_state = [],method="FE"):
	self.time_elapsed = 0   
	if method=="FE": self.integrate=self.forwardEuler
	self.state = np.asfarray(init_state)
    
    def forwardEuler(self,u,f,dt):
	return u+f*dt

    def timestep(self, dt):
	self.state=self.integrate(self.state, self.RHS, dt)
	self.time_elapsed=+dt
    
    def buildRHS(self, I):
	pass
    
class HoHuxNeuron(Neuron):
    """
    Class for the pyramidal cell model of Olufsen et al. (2003)
    The init_state is [v0, h0, n0, s0]
    where
    v0, h0, n0, s0   : are the initial conditions for voltage, h-gating, n-gating and synapse
    """

    def __init__(self,init_state,method="FE"):
	Neuron.__init__(self,init_state,method)
        
    def timestep(self, dt, I):
        #[kappa, a, epsilon, gamma, b, sigma] = self.params
        [v, h, n, s] = self.state #s: synapse function
        
        #capacitance density (muF / cm^(2)), conductance densities (mS/cm^(2)), current densities (muA/cm^(2))
        #and reversal potentials (mV)
        C = 1.	
        g_Na = 100.
        g_K = 80.
        g_L = 0.1
        v_Na = 50.
        v_K = -100.
        v_L = -67.
       #rate constants 
        def a_m(x):
	  return (0.32*(x + 54.)/(1-np.exp(-(x + 54)/4.)))
	  
	def b_m(x):
	  return (0.28*(x + 27.))/(np.exp((x + 27)/5.) - 1.)
	  
	def a_h(x):
	  return 0.128 * np.exp(-(x + 50.)/18.)
	  
	def b_h(x):
	  return 4./(1. + np.exp(-(x + 27.)/5.))
	  
	def a_n(x):
	  return (0.032*(x + 52.)/(1. -np.exp(-(x + 52.)/5.)))
	  
	def b_n(x):
	  return 0.5 * np.exp(-(x + 57.)/40.)
	
	#infinite conductances
	def m_inf(x):
	  return a_m(x)/(a_m(x) + b_m(x))
	  
	def h_inf(x):
	  return a_h(x)/(a_h(x) + b_h(x))
	  
	def n_inf(x):
	  return a_n(x)/(a_n(x) + b_n(x))
	  
	#time constants
	
	def tau_h(x):
	  return 1./(a_h(x) + b_h(x))
	  
	def tau_n(x):
	  return 1./(a_n(x) + b_n(x))
	 
        #dynamical variables
        #v
        self.state[0] = v + (1./C)*(g_Na * m_inf(v)**(3)*h * (v_Na- v) + g_K * n**(4)*(v_K - v) + g_L * (v_L -v) + I) * dt
       
        #h
        self.state[1] = h + ((h_inf(v)-h)/(tau_h(v)))*dt
        
        #n
        self.state[2] = n + ((n_inf(v)-n)/(tau_n(v)))*dt
        
        #synaptic variable, set time constant in accordance with papers
        tau_R = 0.2
        tau_D = 20.
        
        self.state[3] = s + (((1.+np.tanh(v/4.))/2.) * ((1.-s)/tau_R) -s/tau_D) * dt
        #print s
        
        self.time_elapsed += dt

class OLMneuron(Neuron):
    """
    Class for an oriens lacunosum-moleculare interneuron (Tort et al. 2007)
    The init_state is [v0, m0, h0, n0, a0, b0, r0, s0]
    where
    v0, m0, h0, n0, a0, b0,r0, s0   : are the initial conditions for voltage, (m,h,n,a,b,r)-gating and synapse
    """

    def __init__(self,init_state,method="FE"):
        Neuron.__init__(self,init_state,method)
               
        #capacitance density (muF / cm^(2)), conductance densities (mS/cm^(2)), current densities (muA/cm^(2))
        #and reversal potentials (mV
        
       #rate constants 
        def a_m(x):
	  return (-0.1*(x + 38.)/(np.exp(-(x + 38)/10.)-1.))
	  
	def b_m(x):
	  return 4.*np.exp(-(x + 65.)/18.)
	  
	def a_h(x):
	  return 0.07 * np.exp(-(x + 63.)/20.)
	  
	def b_h(x):
	  return 1./(1. + np.exp(-(x + 33.)/10.))
	  
	def a_n(x):
	  return (0.018*(x - 25.)/(1. -np.exp(-(x -25.)/25.)))
	  
	def b_n(x):
	  return (0.0036*(x - 35.)/(np.exp((x -35.)/12.) -1.))
	
	#infinite conductances and time constants
	
	def a_inf(x):
	  return 1./(1. +np.exp(-(x + 14)/16.6))

	def tau_a(x):
	  return 5.
	  
	def b_inf(x):
	  return 1./(1. +np.exp((x + 71)/7.3))
	  
	def tau_b(x):
	  return 1./((0.000009)/(np.exp((x-26.)/18.5)) + ((0.014)/(0.2 + np.exp( -(x + 70.)/11.))))
	  
        def r_inf(x):
	  return 1./(1. + np.exp((x + 84.)/10.2))
	  
	def tau_r(x):
	  return 1./(np.exp(-14.59-0.086*x) + np.exp(-1.87 + 0.0701*x))
	 
	def m_inf(x):
	  return a_m(x)/(a_m(x) + b_m(x))
	  
	def h_inf(x):
	  return a_h(x)/(a_h(x) + b_h(x))
	  
	def n_inf(x):
	  return a_n(x)/(a_n(x) + b_n(x))
	  
	#time constants

	def tau_m(x):
	  return 1./(a_m(x) + b_m(x))
	
	def tau_h(x):
	  return 1./(a_h(x) + b_h(x))
	  
	def tau_n(x):
	  return 1./(a_n(x) + b_n(x))
	 
	  
	self.functions = [a_m, b_m, a_h, b_h, a_n, b_n, a_inf, tau_a, b_inf, tau_b, r_inf, tau_r, m_inf, h_inf, n_inf, tau_m,
	
	                  tau_h, tau_n]                  
        C = 1.3	
        g_L = 0.05
        g_Na = 30.
        g_K = 23.
        g_a = 16.
        g_h =12.
        ###
        v_Na = 90.
        v_K = -100.
        v_a = -90.
        v_h = -32.9
        v_L = -70.
        tau_R = 0.2
        tau_D = 20.
	
        self.params =np.asarray([C,g_L, g_Na, g_K, g_a, g_h, v_Na, v_K, v_a, v_h, v_L, tau_R, tau_D],dtype = 'float')
        
    def buildRHS(self, I):

        [C,g_L, g_Na, g_K, g_a, g_h, v_Na, v_K, v_a, v_h, v_L, tau_R, tau_D] = self.params
        
        [a_m, b_m, a_h, b_h, a_n, b_n, a_inf, tau_a, b_inf, tau_b, r_inf, tau_r, m_inf, h_inf, n_inf, tau_m, tau_h, tau_n]= self.functions
        
        [v, m, h, n, a, b, r, s] = self.state
       
	f0=(1./C)*(g_Na * m**(3)*h * (v_Na- v) + g_K * n**(4)*(v_K - v) + g_a*a*b*(v_a-v) + g_h*r*(v_h-v) + g_L * (v_L -v) + I) 
	f1=(m_inf(v)-m)/tau_m(v)
	f2=(h_inf(v)-h)/tau_h(v)
	f3=(n_inf(v)-n)/tau_n(v)
	f4=(a_inf(v)-a)/tau_a(v)
	f5=(b_inf(v)-b)/tau_b(v)
	f6=(r_inf(v)-r)/tau_r(v)
	f7=(1.+np.tanh(v/4.))/2.* (1.-s)/tau_R-s/tau_D
	
	self.RHS=np.asfarray([f0,f1,f2,f3,f4,f5,f6,f7])



class O_Rotstein(Neuron):
    """
    Class for an oriens lacunosum-moleculare interneuron (Rotstein et al. 2005)
    The init_state is [v0, m0, h0, n0, p0, hf0, hs0, s0 ]
    where
    v0, m0, h0, n0, p0, hf0, hs0, s0   : are the initial conditions for voltage, (m,h,n,p, hf, hs)-gating and synapse
    """

    def __init__(self,init_state,method="FE"):
        Neuron.__init__(self,init_state,method)
               
        #capacitance density (muF / cm^(2)), conductance densities (mS/cm^(2)), current densities (muA/cm^(2))
        #and reversal potentials (mV
        
       #rate constants 
    def a_m(x):
	  return (-0.1*(x + 23.)/(np.exp(-(x + 23.)/10.)-1.))
	  
	def b_m(x):
	  return 4.*np.exp(-(x + 48.)/18.)
	  
	def a_h(x):
	  return 0.07 * np.exp(-(x + 37.)/20.)
	  
	def b_h(x):
	  return 1./(1. + np.exp(-(x + 7.)/10.))
	  
	def a_n(x):
	  return -0.01*(x + 27.)/(np.exp(-(x  + 27.)/10.) -1.)
	  
	def b_n(x):
	  return 0.125*np.exp(-(x + 37.)/80.)
	  
	def a_p(x):
	  return 1./(0.15*(1 + np.exp(-(x + 38.)/6.5)))
	  
	def b_p(x):
	  return np.exp(-(x + 38.)/6.5)/(0.15*(1 + np.exp(-(x+ 38.)/6.5)))
	
	#infinite conductances and time constants
	
	
	def hf_inf(x):
	  return 1./(1. + np.exp((x + 79.2)/(9.78)))
	  
    def tau_hf(x):
	  return 0.51/(np.exp((x-1.7)/10.) + np.exp(-(x + 340.)/52.)) + 1.
	  
	def hs_inf(x):
	  return 1./(1. + np.exp((x + 2.83)/15.9))**(58.)
	  
    def tau_hs(x):
	  return 5.6/(np.exp((x - 1.7)/14.) + np.exp(-(x + 260.)/43.)) + 1.
	  
	def m_inf(x):
	  return a_m(x)/(a_m(x) + b_m(x))
	  
	def h_inf(x):
	  return a_h(x)/(a_h(x) + b_h(x))
	  
	def n_inf(x):
	  return a_n(x)/(a_n(x) + b_n(x))
          
    def p_inf(x):
        return a_p(x)/(a_p(x) + b_p(x))
	  
	#time constants

	def tau_m(x):
          return 1./(a_m(x) + b_m(x))
	
	def tau_h(x):
          return 1./(a_h(x) + b_h(x))
	  
	def tau_n(x):
        return 1./(a_n(x) + b_n(x))
        
    def tau_p(x):
        return 1./(a_p(x) + b_p(x))
	 
	  
	self.functions = [a_m, b_m, a_h, b_h, a_n, b_n, a_p, b_p, hf_inf, tau_hf, hs_inf, tau_hs, m_inf, h_inf, n_inf, p_inf, tau_m, tau_h, tau_n, tau_p]
        
        C = 1.0
        g_L = 0.5
        g_Na = 52.
        g_K = 11.
        g_p = 0.5
        g_h =12.
        ###
        v_Na = 55.
        v_K = -90.
        v_h = -20
        v_L = -65.
        
        #I_app = -1.8
	
    self.params =np.asarray([C,g_L, g_Na, g_K, g_p, g_h, v_Na, v_K, v_h, v_L],dtype = 'float')
        
    def buildRHS(self, I):

        [C,g_L, g_Na, g_K, g_p, g_h, v_Na, v_K, v_h, v_L] = self.params
        
        [a_m, b_m, a_h, b_h, a_n, b_n, a_p, b_p, hf_inf, tau_hf, hs_inf, tau_hs, m_inf, h_inf, n_inf, p_inf, tau_m, tau_h, tau_n, tau_p] = self.functions
        
        [v, m, h, n, p, hf, hs, s] = self.state
       
	f0=(1./C)*(g_Na * m**(3)*h * (v_Na- v) + g_K * n**(4)*(v_K - v) + g_L * (v_L -v) + g_p*p*(v_Na-v) + g_h*(0.65*hf + 0.35*hs)*(v_h-v) + I)
        
	f1=(m_inf(v)-m)/tau_m(v)
	f2=(h_inf(v)-h)/tau_h(v)
	f3=(n_inf(v)-n)/tau_n(v)
	f4=(p_inf(v)-p)/tau_p(v)
	f5=(hf_inf(v)-hf)/tau_hf(v)
	f6=(hs_inf(v)-hs)/tau_hs(v)
    
    #synaptic variable
    alpha_O = 5.
    beta_O = 0.05
    f7=(alpha_O/2.)*((1.+np.tanh(v/0.1)))* (1.-s)-beta_O*s
	
	self.RHS=np.asfarray([f0,f1,f2,f3,f4,f5,f6,f7])

class I_Rotstein(Neuron):
    """
        Class for a fast-spiking (FS) interneuron (Rotstein et al. 2005)
        The init_state is [v0, m0, h0, n0, s0 ]
        where
        v0, m0, h0, n0, s0   : are the initial conditions for voltage, (m,h,n)-gating and synapse
        """
    
    def __init__(self,init_state,method="FE"):
        Neuron.__init__(self,init_state,method)
    
    #capacitance density (muF / cm^(2)), conductance densities (mS/cm^(2)), current densities (muA/cm^(2))
    #and reversal potentials (mV
    
    #rate constants
    def a_m(x):
        return (0.32*(x + 54.)/(1.-np.exp(-(x + 54.)/4.)))
            
    def b_m(x):
        return 0.28*(x + 27.)/(np.exp((x + 27.)/5.) -1.)
            
    def a_h(x):
        return 0.128 * np.exp(-(x + 50.)/18.)
            
    def b_h(x):
        return 4./(1. + np.exp(-(x + 27.)/5.))
            
    def a_n(x):
        return (0.032*(x + 52.)/(1.-np.exp(-(x  + 52.)/5.)))
            
    def b_n(x):
        return 0.5*np.exp(-(x + 57.)/40.)
            
    #infinite conductances and time constants
            
    def m_inf(x):
        return a_m(x)/(a_m(x) + b_m(x))
            
    def h_inf(x):
        return a_h(x)/(a_h(x) + b_h(x))
            
    def n_inf(x):
        return a_n(x)/(a_n(x) + b_n(x))
        
    #time constants
        
    def tau_m(x):
        return 1./(a_m(x) + b_m(x))
        
    def tau_h(x):
        return 1./(a_h(x) + b_h(x))
        
    def tau_n(x):
        return 1./(a_n(x) + b_n(x))

        
     self.functions = [a_m, b_m, a_h, b_h, a_n, b_n, m_inf, h_inf, n_inf, tau_m, tau_h, tau_n]
        
        C = 1.0
        g_L = 0.1
        g_Na = 100.
        g_K = 80.
        g_h =1.46
        ###
        v_Na = 50.
        v_K = -100.
        v_h = -20
        v_L = -67.
        tau_R = 0.2
        tau_D = 20.
        
        #I_app = 0.48
        
        self.params =np.asarray([C,g_L, g_Na, g_K, g_h, v_Na, v_K, v_h, v_L, tau_R, tau_D],dtype = 'float')
    
    def buildRHS(self, I):
        
        [C,g_L, g_Na, g_K, g_p, g_h, v_Na, v_K, v_h, v_L, tau_R, tau_D] = self.params
        
        [a_m, b_m, a_h, b_h, a_n, b_n, m_inf, h_inf, n_inf, tau_m, tau_h, tau_n]] = self.functions
        
        [v, m, h, n, s] = self.state
        
        f0=(1./C)*(g_Na * m**(3)*h * (v_Na- v) + g_K * n**(4)*(v_K - v) + g_L * (v_L -v) + I)
        f1=(m_inf(v)-m)/tau_m(v)
        f2=(h_inf(v)-h)/tau_h(v)
        f3=(n_inf(v)-n)/tau_n(v)
        
        #synaptic variable
        alpha_I = 15.
        beta_I = 0.11
        f4=(alpha_I/2.)*((1.+np.tanh(v/0.1)))* (1.-s)-beta_I*s
        
        self.RHS=np.asfarray([f0,f1,f2,f3,f4])



    