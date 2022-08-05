import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

import os
import time
import configparser

class loading_curve_fitting():

    def __init__(self,loading_curve,n1trust,n2trust=0):
        self.loading_curve = loading_curve
        self._n1_trust_radius = np.float64(n1trust)
        self._n2_trust_radius = np.float64(n2trust)
        self._R = np.float64(1)
        self._g = np.float64(1)
        self._b = np.float64(1)
        self._cov_matrix = np.empty(1)
        self._R_cov = np.float64(1)
        self._g_cov = np.float64(1)
        self._b_cov = np.float64(1)
        self._popscale = np.float64(1.023*1e8)
        self._trash_bin = 0
        
    def n1(self,t,R):
        """Linear approximation for the trapped population.
        
        Parameters
        ----------
        t : float
            Instant at which to calculate the trapped
            population.
            
        R : float
            Loading rate for the MOT.
        
        Returns
        -------
        n1 : float
            Trapped population calculated at the given
            instant with the linear approximation
            using the given value of L.
        """
        
        n1 = R*t
        return n1

    def n2(self,t,R,g):
        """Solution for the simple model.
        
        Parameters
        ----------
        t : float
            Instant at which to calculate the trapped
            population.
            
        R : float
            Loading rate for the MOT.
            
        g: float
            Rate of loss from the trap due to collisions
            between trapped and free atoms.    
        
        Returns
        -------
        n2 : float
            Trapped population calculated at the given
            instant with the exponential approximation
            using the given values of R and g.
        """
        
        n2 = (R/g)*(1-np.exp(-g*t))
        return n2
    
    def n3(self,t,R,g,b):
        """Solution for the intratrap collision
        model.
        
        Parameters
        ----------
        t : float
            Instant at which to calculate the trapped
            population.
            
        R : float
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
            using the given values of R, g and b.
        """
        
        D = np.sqrt(g**2 + 4*R*b)
        n3 = D/(2*b)*np.tanh(D*t/2 + np.arctanh(g/D)) - g/(2*b)
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
        with the solution for the simple model. If 
        n2_trust_radius is given as 0, it us under-
        stood that the simple solution is the one to
        be used, and thus this simply returns the
        entire curve,
        
        Returns
        -------
        n2_fitting_data : numpy array
            Array of loading curve points from instant
            0 to n2trust.
        """
        
        if self._n2_trust_radius == 0:
            return self.loading_curve
        
        else:        
            data_point = 1
            n2_fitting_data = np.array([self.loading_curve[0]])
            while self.loading_curve[data_point,0] <= self._n2_trust_radius:
                n2_fitting_data = np.append(n2_fitting_data,[self.loading_curve[data_point]],axis=0)
                data_point += 1
            return n2_fitting_data
    
    def _n3_fitting_data(self):
        """Selects the points to be used for fitting
        with the solution for the intratrap collisions
        model. This is always the entire curve.
        
        Returns
        -------
        loading_curve : numpy array
            Points from the entire loading curve.
        """
        return self.loading_curve

    def _n1_fitting(self):
        """Fits the linear approximation to its
        approppriate fitting interval and obtains
        R.
        
        Returns
        -------
        fitL : float
            Fitted value of R.
            
        fitL_cov : float
            Estimated covariance of R from the fitting.
        """
        
        n1_fitting_data = self._n1_fitting_data()
        fitR, fitR_cov = optimize.curve_fit(self.n1, n1_fitting_data[:,0], n1_fitting_data[:,1])
        return fitR, fitR_cov
        
    def _n2_fitting(self,R=None):
        """Fits the exponential approximation to its
        approppriate fitting interval and obtains g.
        If a value of R is given, fits R and g 
        simultaneously and obtains both.
        
        Parameters
        ----------
        R : float
            If none is given, fits only g. If any value
            is given, simultaneously fits R and g using 
            that value as a starting guess.
        
        Returns
        -------
        fitRg : float
            Fitted value of g or both R and g.
            
        fitRg_cov : float
            Estimated covariance of g or R and g
            from the fitting.
        """
        
        n2_fitting_data = self._n2_fitting_data()
        
        if R == None:
            n2 = lambda t, g: self.n2(t,self._R,g)
            fitRg, fitRg_cov = optimize.curve_fit(n2, n2_fitting_data[:,0], n2_fitting_data[:,1],bounds=(0,np.inf))
        
        else:
            fitRg, fitRg_cov = optimize.curve_fit(self.n2, n2_fitting_data[:,0], n2_fitting_data[:,1],p0=[self._R,self._g],bounds=(0,np.inf))
        
        return fitRg, fitRg_cov        
        
        
    def _n3_fitting(self,g=None):
        """Fits the exact solution to the whole curve, and
        obtains b. If a value of g is given, simultaneously
        fits g and b, and returns both.
        
        Parameters
        ----------
            
        g : float
            If none is given, fits only b. If any value is given,
            fits g and b simultaneously.
            
        Returns
        -------
        fitgb : float
            Fitted values of b or both g and b.
            
        fitgb_cov: float
            Estimated covariances from the fitting of either
            b or both g and b.
        """
        
        n3_fitting_data = self._n3_fitting_data()
            
        if g == None:
            n3 = lambda t, b: self.n3(t,self._R,self._g,b)
            fitgb, fitgb_cov = optimize.curve_fit(n3, n3_fitting_data[:,0], n3_fitting_data[:,1],bounds=(0,np.inf))
            
        else:
            n3 = lambda t, g, b: self.n3(t,self._R,g,b)
            fitgb, fitgb_cov = optimize.curve_fit(n3, n3_fitting_data[:,0], n3_fitting_data[:,1],p0=[self._g,self._b],bounds=([0,0],[np.inf,np.inf]))
        
        return fitgb, fitgb_cov        
    
        
    def fitting_routine_a(self):
        """Executes a fitting routine for the parameters R and g.
        Fits R to the interval t<n1_trust_radius, then fits g over
        the entire curve with R fixed. After that, a simultaneous fit
        of both R and g is performed over the entire curve using the 
        previous results as starting guesses.
        """
        
        [self._R], self._cov_matrix = self._n1_fitting()
        self._R_cov = np.sqrt(np.diag(self._cov_matrix))
        
        [self._g], self._cov_matrix = self._n2_fitting()
        self._g_cov = np.sqrt(np.diag(self._cov_matrix))
        
        [self._R,self._g], self._cov_matrix = self._n2_fitting(R=self._R)
        [self._R_cov,self._g_cov] = np.sqrt(np.diag(self._cov_matrix))

        return None
    
    def fitting_routine_b(self):
        """Executes a fitting routine over the three
        available solutions n1, n2 and n3. Fits each
        parameter individually, with collective fitting
        in between, using the latest fitted parameters 
        as initial guesses. 
        """
        
        #not necessary for now
        
        return None
    
    def fit_a(self):
        """Performs the fitting routine for the
        simple model and returns the fitted parameters
        alongside their calculated covariances.
        """

        self.fitting_routine_a()
        
        param_list = np.array([self._R,self._g])
        cov_list   = np.array([self._R_cov,self._g_cov])
        
        return param_list, cov_list
    
    def fit_b(self):
        """Performs the fitting routine for the
        intratrap collisions model and returns the
        fitted parameters alongside their calculated
        covariances.
        """
        
        #self.fitting_routine_b()
        
        #param_list = np.array([self._R,self._g,self._b])
        #cov_list   = np.array([self._R_cov,self._g_cov,self._b_cov])
        
        #refer to fitting_routine_b
        
        return None
    
    def fitting_plot(self):
        """Plots the fitted loading curve over the
        scatter plot of the experimental points.
        """
        
        plt.xlabel('Tempo (s)')
        plt.ylabel('No de átomos')
        plt.scatter(self.loading_curve[:,0],self.loading_curve[:,1]*[self._popscale],alpha=0.5, label='Curva experimental')
        plt.plot(self.loading_curve[:,0], self.n3(self.loading_curve[:,0], self._R, self._g, self._b)*[self._popscale], c='red', label='Ajuste teórico')
        plt.legend(bbox_to_anchor=(1,0.2))
        plt.show()
            
        return None