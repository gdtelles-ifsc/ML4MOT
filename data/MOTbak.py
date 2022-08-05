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
    
    def __init__(self,L,g,b,w):
        self.loading_rate = L
        self.gamma = g
        self.b = b
        self.gaussian_width = w
        self.measurement_times = np.array([])
        self.measurements = np.array([])
        self.noisy_measurements = np.array([])
        
    def _a(self):
        return self.gamma
    
    def _b(self):
        return self.b
    
    def N(self,t):
        a = self._a()
        b = self._b()
        D = np.sqrt(a**2 + 4*self.loading_rate*b)
        return D/(2*b)*np.tanh(D*t/2 + np.arctanh(a/D)) - a/(2*b)
    
    def load_MOT(self,dt,endtime):
        self.measurement_times = np.arange(0,endtime,dt)
        self.measurements = np.array([[t,self.N(t)] for t in self.measurement_times])
        return None
    
    def measure_loading(self):
        nom = len(self.measurements)
        self.noise = np.array([[0,(np.random.random()-0.5)*(10e-3*np.random.random())] for i in list(range(nom))])
        self.noisy_measurements = self.measurements + self.noise
        return None
                            
    def return_loading_curve(self):
        plt.scatter(self.noisy_measurements[:,0],self.noisy_measurements[:,1],s=10,alpha=0.1)
        return self.noisy_measurements
    
class MOTinterface(mli.Interface):
    """M-LOOP interface for a Python-controlled
    magneto-optical trap.
    
    Parameters
    ----------

    """
    
    def __init__(self,simultaneous_fitting=True):
        super(MOTinterface,self).__init__()
        self.loading_curve = np.empty([0,2])
        self._n1_trust_radius = np.float64(1)
        self._n2_trust_radius = np.float64(2)
        self._L = np.float64(1)
        self._g = np.float64(1)
        self._b = np.float64(1)
        self._cov_matrix = np.empty(1)
        self._L_cov = np.float64(1)
        self._g_cov = np.float64(1)
        self._b_cov = np.float64(1)
        self._trash_bin = 0
        self._timescale = np.float64(15.03/1320)
        self._ini_path = 'params.ini'
        self._data_path = 'MOT1pd.txt'
        self._settings = configparser.ConfigParser()
        self._simultaneous_fitting = simultaneous_fitting
        
    def n1(self,t,L):
        return L*t

    def n2(self,t,L,g):
        return (L/g)*(1-np.exp(-g*t))
    
    def n3(self,t,L,g,b):
        a = g
        D = np.sqrt(a**2 + 4*L*b)
        return D/(2*b)*np.tanh(D*t/2 + np.arctanh(a/D)) - a/(2*b)
    
    def _n1_fitting_data(self):
        data_point = 1
        n1_fitting_data = np.array([self.loading_curve[0]])
        while self.loading_curve[data_point,0] <= self._n1_trust_radius:
            n1_fitting_data = np.append(n1_fitting_data,[self.loading_curve[data_point]],axis=0)
            data_point += 1
        return n1_fitting_data
    
    def _n2_fitting_data(self):
        data_point = 1
        n2_fitting_data = np.array([self.loading_curve[0]])
        while self.loading_curve[data_point,0] <= self._n2_trust_radius:
            n2_fitting_data = np.append(n2_fitting_data,[self.loading_curve[data_point]],axis=0)
            data_point += 1
        return n2_fitting_data
    
    def _n3_fitting_data(self):
        return self.loading_curve

    def _n1_fitting(self):
        n1_fitting_data = self._n1_fitting_data()
        fitL, fitL_cov = optimize.curve_fit(self.n1, n1_fitting_data[:,0], n1_fitting_data[:,1])
        return fitL, fitL_cov
        
    def _n2_fitting(self,L=None):
        n2_fitting_data = self._n2_fitting_data()
        
        if L == None:
            n2 = lambda t, g: self.n2(t,self._L,g)
            fitLg, fitLg_cov = optimize.curve_fit(n2, n2_fitting_data[:,0], n2_fitting_data[:,1])
        
        else:
            fitLg, fitLg_cov = optimize.curve_fit(self.n2, n2_fitting_data[:,0], n2_fitting_data[:,1],p0=[self._L,self._g])
        
        return fitLg, fitLg_cov        
        
        
    def _n3_fitting(self,L=None,g=None):
        n3_fitting_data = self._n3_fitting_data()
            
        if g == None:
            n3 = lambda t, b: self.n3(t,self._L,self._g,b)
            fitLgB, fitLgB_cov = optimize.curve_fit(n3, n3_fitting_data[:,0], n3_fitting_data[:,1])
            
        else:
            n3 = lambda t, g, b: self.n3(t,self._L,g,b)
            fitLgB, fitLgB_cov = optimize.curve_fit(n3, n3_fitting_data[:,0], n3_fitting_data[:,1],p0=[self._g,self._b])
        
        return fitLgB, fitLgB_cov        
    
        
    def fitting_routine(self):
        """Executes a fitting routine over the three
        available solutions n1, n2 and n3. Fits each
        parameter individually, with collective fitting
        in between, using the latest fitted parameters 
        as initial guesses.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        
        [self._L], self._cov_matrix = self._n1_fitting()
        self._L_cov = np.sqrt(np.diag(self._cov_matrix))
        
        [self._g], self._cov_matrix = self._n2_fitting()
        self._g_cov = np.sqrt(np.diag(self._cov_matrix))
        
        #Uncomment below for simultaneous fitting of L and g. Not recommended.
        #if self._simultaneous_fitting:
        #    [self._L,self._g], self._cov_matrix = self._n2_fitting(L=self._L)
        #    [self._L_cov,self._g_cov] = np.sqrt(np.diag(self._cov_matrix))
            
        [self._b], self._cov_matrix = self._n3_fitting()
        [self._b_cov] = np.sqrt(np.diag(self._cov_matrix))
        
        if self._simultaneous_fitting:
            [self._g,self._b], self._cov_matrix = self._n3_fitting(g=self._g)
            [self._g_cov,self._b_cov] = np.sqrt(np.diag(self._cov_matrix))
                    
        self.fitting_plot()
        
        print(self._L,self._L_cov)
        print(self._g,self._g_cov)
        print(self._b,self._b_cov)
        
        return None
    
    def fitting_plot(self):
        plt.scatter(self.loading_curve[:,0],self.loading_curve[:,1],alpha=0.5)
        plt.plot(self.loading_curve[:,0], self.n3(self.loading_curve[:,0], self._L, self._g, self._b), c='red')
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
            pass
        
        curve = open(self._data_path,'r')
        curvelines = curve.readlines()[1:]
        curve.close()
        os.remove(self._data_path)
        
        for line in curvelines:
            formatted_line = np.array([np.float64([line.split('\t')[0],line.split('\t')[1][:-2]])])
            self.loading_curve = np.concatenate((self.loading_curve,formatted_line*[self._timescale,1]))
        self.loading_curve -= np.array([[0,self.loading_curve[0,1]]]*len(self.loading_curve))        
        print(self.loading_curve)
        
        for point in self.loading_curve:
            if point[1]<0:
                print(point)
        
        
        self.fitting_routine()
        
        cost = self._L
        uncer = 1e-3
        bad = False
    
        cost_dict = {'cost' : cost, 'uncer' : uncer, 'bad' : bad}

        return cost_dict 