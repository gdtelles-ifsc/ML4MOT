import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

import os
import time
import configparser

class MOTexperiment():
    """Generates loading curves from the solution considering
    losses from two-body collisions within the trap and between
    one trapped and one free atom.
    
    Parameters
    ----------
    L : float
        L parameter in the solution. Gives the rate at which 
        atoms are captured by the trap.
        
    gamma : float
        Gamma in the solution, which is equal to a. Gives 
        the rate at which atoms are lost due to collisions
        within the trap.
        
    b: float
        b in the solution, defined in terms of beta and 
        the gaussian width. Beta gives the rate at which
        atoms are lost due to collisions with atoms
        outside the trap.
        
    w: float
        Gaussian width of the trap. Determined 
        experimentally.
    """
    
    def __init__(self,L,g,b,w):
        self.loading_rate = L
        self.gamma = g
        self.b = b
        self.gaussian_width = w
        self.measurement_times = np.array([])
        self.measurements = np.array([])
        self.noisy_measurements = np.array([])
        
    def _a(self):
        """Function to set a as a variable.
        
        Returns
        -------
        gamma : float
            Parameter gamma, which is equal to a.
        """
        
        return self.gamma
    
    def _b(self):
        """Function to set b as a variable.
        
        Returns
        -------
        b : float
            Parameter b
        """
        
        return self.b
    
    def N(self,t):
        """Applies the solution of the charging curve to
        compute the trapped population at a given time.
        
        Parameters
        ----------
        t : float
            Time at which to calcualte the population
            
        Returns
        -------
        N : float
            Calculated trapped population
        """
        
        a = self._a()
        b = self._b()
        D = np.sqrt(a**2 + 4*self.loading_rate*b)
        N = D/(2*b)*np.tanh(D*t/2 + np.arctanh(a/D)) - a/(2*b)
        return N
    
    def load_MOT(self,dt,endtime):
        """Calls for the computation of trapped population
        at intervals dt from instant 0 to a given final time
        of measurement.
        
        Parameters
        ----------
        dt : float
            Time interval between measurements, i.e., computations 
            of the trapped population.
            
        endtime : float
            Last point in time at which the population is to be 
            computed.
        """
        
        self.measurement_times = np.arange(0,endtime,dt)
        self.measurements = np.array([[t,self.N(t)] for t in self.measurement_times])
        return None
    
    def measure_loading(self):
        """Adds to noise to simulate a real loading curve
        with the data generated from the solution.
        """
        nom = len(self.measurements)
        self.noise = np.array([[0,(np.random.random()-0.5)*(10e-3*np.random.random())] for i in list(range(nom))])
        self.noisy_measurements = self.measurements + self.noise
        return None
                            
    def return_loading_curve(self):
        """Plots the points of the loading curve with added
        noise and returns the curve points.
        
        Returns
        -------
        noisy_measurements : numpy array
            Points of the loading curve with random noise 
            added.
        """
        
        plt.scatter(self.noisy_measurements[:,0],self.noisy_measurements[:,1],s=10,alpha=0.1)
        return self.noisy_measurements
    
class MOTinterface(mli.Interface):
    """M-LOOP interface for a Python-controlled
    magneto-optical trap.
    
    Parameters
    ----------
    n1trust : float
        Superior limit of the time interval - starting at
        0 - where the linear approximation will be used for
        fitting.
        
    n2trust : float
        Superior limit of the time interval - starting at
        0 - where the exponential approximation will be
        used for fitting.
        
    simultaneous_fitting : boolean
        Sets whether g and b will be simultaenously fitted
        as part of the fitting routine or not.
    """
    
    def __init__(self,n1trust,n2trust,azero=False,bzero=False):
        super(MOTinterface,self).__init__()
        self.loading_curve = np.empty([0,2])
        self._n1_trust_radius = np.float64(n1trust)
        self._n2_trust_radius = np.float64(n2trust)
        self._L = np.float64(1)
        self._g = np.float64(1)
        self._b = np.float64(1)
        self._cov_matrix = np.empty(1)
        self._L_cov = np.float64(1)
        self._g_cov = np.float64(1)
        self._b_cov = np.float64(1)
        self._trash_bin = 0
        self._timescale = np.float64(15.03/1320)
        self._ini_path = '../data/params.ini'
        self._data_path = '../data/MOT1pd.txt'
        self._settings = configparser.ConfigParser()
        self._a_is_zero = azero
        self._b_is_zero = bzero
        self._popscale = np.float64(1.023*1e8)
        
    def n1(self,t,L):
        """Linear approximation for the trapped population.
        
        Parameters
        ----------
        t : float
            Instant at which to calculate the trapped
            population.
            
        L : float
            Loading rate for the MOT.
        
        Returns
        -------
        n1 : float
            Trapped population calculated at the given
            instant with the linear approximation
            using the given value of L.
        """
        
        n1 = L*t
        return n1

    def n2(self,t,L,g):
        """Exponential approximation for the trapped
        population.
        
        Parameters
        ----------
        t : float
            Instant at which to calculate the trapped
            population.
            
        L : float
            Loading rate for the MOT.
            
        g: float
            Rate of loss from the trap due to collisions
            between trapped and free atoms.    
        
        Returns
        -------
        n2 : float
            Trapped population calculated at the given
            instant with the exponential approximation
            using the given values of L and g.
        """
        
        n2 = (L/g)*(1-np.exp(-g*t))
        return n2
    
    def n3(self,t,L,g,b):
        """Exact solution for the trapped
        population.
        
        Parameters
        ----------
        t : float
            Instant at which to calculate the trapped
            population.
            
        L : float
            Loading rate for the MOT.
            
        g: float
            Rate of loss from the trap due to collisions
            between trapped and free atoms.    
            
        b : float
            Factor related to the constant beta, which 
            gives the rate of loss due to collisions 
            between trapped atoms, and the 
            gaussian width of the trap.
        
        Returns
        -------
        n3 : float
            Trapped population calculated at the given
            instant with the exact solution
            using the given values of L, g and b.
        """
        
        a = g
        D = np.sqrt(a**2 + 4*L*b)
        n3 = D/(2*b)*np.tanh(D*t/2 + np.arctanh(a/D)) - a/(2*b)
        return n3
    
    def _n1_fitting_data(self):
        """Selects the points to be used for fitting
        with the linear approximation.
        
        Returns
        -------
        n1_fitting_data : numpy array
            Array of loading curve points from instant
            0 to n1trust.
        """
        
        data_point = 1
        n1_fitting_data = np.array([self.loading_curve[0]])
        while self.loading_curve[data_point,0] <= self._n1_trust_radius:
            n1_fitting_data = np.append(n1_fitting_data,[self.loading_curve[data_point]],axis=0)
            data_point += 1
        return n1_fitting_data
    
    def _n2_fitting_data(self):
        """Selects the points to be used for fitting
        with the exponential approximation.
        
        Returns
        -------
        n2_fitting_data : numpy array
            Array of loading curve points from instant
            0 to n2trust.
        """
        data_point = 1
        n2_fitting_data = np.array([self.loading_curve[0]])
        while self.loading_curve[data_point,0] <= self._n2_trust_radius:
            n2_fitting_data = np.append(n2_fitting_data,[self.loading_curve[data_point]],axis=0)
            data_point += 1
        return n2_fitting_data
    
    def _n3_fitting_data(self):
        """Selects the points to be used for fitting
        with the exact solution, which are simply
        the entire curve.
        
        Returns
        -------
        loading_curve : numpy array
            Points from the entire loading curve.
        """
        return self.loading_curve

    def _n1_fitting(self):
        """Fits the linear approximation to its
        approppriate fitting interval and obtains
        L.
        
        Returns
        -------
        fitL : float
            Fitted value of L.
            
        fitL_cov : float
            Estimated covariance of L from the fitting.
        """
        
        n1_fitting_data = self._n1_fitting_data()
        fitL, fitL_cov = optimize.curve_fit(self.n1, n1_fitting_data[:,0], n1_fitting_data[:,1])
        return fitL, fitL_cov
        
    def _n2_fitting(self,L=None):
        """Fits the exponential approximation to its
        approppriate fitting interval and obtains g.
        If a value of L is given, fits L and g 
        simultaneously and obtains both.
        
        Parameters
        ----------
        L : float
            If none is given, fits only g. If any value
            is given, simultaneously fits L and g.
        
        Returns
        -------
        fitLg : float
            Fitted value of L or both L and g.
            
        fitLg_cov : float
            Estimated covariance of L or L and g
            from the fitting.
        """
        
        n2_fitting_data = self._n2_fitting_data()
        
        if L == None:
            n2 = lambda t, g: self.n2(t,self._L,g)
            fitLg, fitLg_cov = optimize.curve_fit(n2, n2_fitting_data[:,0], n2_fitting_data[:,1],bounds=(0,np.inf))
        
        else:
            fitLg, fitLg_cov = optimize.curve_fit(self.n2, n2_fitting_data[:,0], n2_fitting_data[:,1],p0=[self._L,self._g])
        
        return fitLg, fitLg_cov        
        
        
    def _n3_fitting(self,L=None,g=None):
        """Fits the exact solution to the whole curve, and
        obtains b. If a value of g is given, simultaneously
        fits g and b, and returns both.
        
        Parameters
        ----------
        L : float
            This parameter isn't used even if a value is given.
            It's only here in case of a future expansion of the
            code where simultaneously fitting of L, g and b is
            used.
            
        g : float
            If none is given, fits only b. If any value is given,
            fits g and b simultaneously.
            
        Returns
        -------
        fitLgb : float
            Fitted values of b or both g and b.
            
        fitLgb_cov: float
            Estimated covariances from the fitting of either
            b or both g and b.
        """
        
        n3_fitting_data = self._n3_fitting_data()
            
        if g == None:
            n3 = lambda t, b: self.n3(t,self._L,self._g,b)
            fitLgB, fitLgB_cov = optimize.curve_fit(n3, n3_fitting_data[:,0], n3_fitting_data[:,1],bounds=(0,np.inf))
            
        else:
            n3 = lambda t, g, b: self.n3(t,self._L,g,b)
            fitLgB, fitLgB_cov = optimize.curve_fit(n3, n3_fitting_data[:,0], n3_fitting_data[:,1],p0=[self._g,self._b],bounds=([0,0],[self._g,np.inf]))
        
        return fitLgB, fitLgB_cov        
    
        
    def fitting_routine(self):
        """Executes a fitting routine over the three
        available solutions n1, n2 and n3. Fits each
        parameter individually, with collective fitting
        in between, using the latest fitted parameters 
        as initial guesses. Currently fits n1 individually,
        then g and individually, then b individually and
        finally g and b simultaneously. There is a commented
        section of code that can be uncommented for simultaneous
        fitting of L and g.
        """
        
        [self._L], self._cov_matrix = self._n1_fitting()
        self._L_cov = np.sqrt(np.diag(self._cov_matrix))
        
        if self._a_is_zero:
            self._g = np.float64(0)
            self._g_cov = np.float64(0)
        else:
            [self._g], self._cov_matrix = self._n2_fitting()
            self._g_cov = np.sqrt(np.diag(self._cov_matrix))
        
        if self._b_is_zero:
            self._b = np.float64(0)
            self._b_cov = np.float64(0)
        else:
            [self._b], self._cov_matrix = self._n3_fitting()
            [self._b_cov] = np.sqrt(np.diag(self._cov_matrix))
        
        if not self._a_is_zero and not self._b_is_zero:
            [self._g,self._b], self._cov_matrix = self._n3_fitting(g=self._g)
            [self._g_cov,self._b_cov] = np.sqrt(np.diag(self._cov_matrix))
                
        self.fitting_plot()
        
        print(self._L,self._L_cov)
        print(self._g,self._g_cov)
        print(self._b,self._b_cov)
        
        return None
    
    def fitting_plot(self):
        """Plots the fitted loading curve over the
        scatter plot of the experimental points.
        """
        
        plt.xlabel('Tempo (s)')
        plt.ylabel('No de átomos')
        plt.scatter(self.loading_curve[:,0],self.loading_curve[:,1]*[self._popscale],alpha=0.5, label='Curva experimental')
        plt.plot(self.loading_curve[:,0], self.n3(self.loading_curve[:,0], self._L, self._g, self._b)*[self._popscale], c='red', label='Ajuste teórico')
        plt.legend(bbox_to_anchor=(1,0.2))
        plt.show()
            
        return None
        
        
    def get_next_cost_dict(self,params_dict):
        """Receives the latest set of optimized parameters
        from a dictionary, runs the experiment and updates
        a cost dictionary with the resulting cost, 
        uncertainty, if available, and with a "bad" label,
        if necessary.
        
        Parameters
        ----------
        params_dict : dictionary
            Holds as an a element a list of the best
            parameters from the most recent update of the
            internal model.
        
        Returns
        -------
        cost_dict : dictionary
            Holds elements for the cost and uncertainty
            values, plus the 'bad' flag. For bad runs, only
            the flag need be passed. Otherwise, the
            uncertainty remains optional.
        """
        params = params_dict['params']
        
        coolingfrequency = str(params[0])
        zcompensation = str(params[1])
        pushfrequency = str(params[2])
        
        self._settings.read(self._ini_path)
        
        self._settings['MOT']['coolingfrequency'] = coolingfrequency
        self._settings['MOT']['zcompensation'] = zcompensation
        self._settings['MOT']['pushfrequency'] = pushfrequency
        
        #at this point, the lab software has to somehow know that the
        #params.ini file has been updated and run the experiment. 
        #If there's anything that needs to be done on the M-LOOP side
        #for this to happen, this is where that would go.
        
        time.sleep(15)
        
        while not os.path.exists(self._data_path):
            time.sleep(5)
            print('Curve file not found, waiting...')
            pass
        
        curve = open(self._data_path,'r')
        curvelines = curve.readlines()[1:]
        curve.close()
        os.remove(self._data_path)
        
        for line in curvelines:
            formatted_line = np.array([np.float64([line.split('\t')[0],line.split('\t')[1][:-2]])])
            self.loading_curve = np.concatenate((self.loading_curve,formatted_line*[self._timescale,1]))
        self.loading_curve -= np.array([self.loading_curve[0]]*len(self.loading_curve))        

        self.fitting_routine()
        
        cost = self._L
        uncer = 1e-3
        bad = False
    
        cost_dict = {'cost' : cost, 'uncer' : uncer, 'bad' : bad}

        return cost_dict 