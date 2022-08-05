import numpy as np

import mloop.interfaces as mli

import cleaner
import curvefit

import os
import time
import configparser

class MOTinterface(mli.Interface):
    """M-LOOP interface for a Python-controlled
    magneto-optical trap.
    
    Parameters
    ----------
    params_path : string
        Path of the params.ini file where the configurations
        for the MOT are set.
        
    data_path : string
        Path of the MOTpd.txt file that contains the loading
        curve data.
        
    n1trust : float
        Superior limit of the time interval - starting at
        0 - where the linear approximation will be used for
        fitting.
        
    n2trust : float
        Superior limit of the time interval - starting at
        0 - where the exponential solution will be used
        for fitting. If 0, the exponential solution is
        used for the whole curve.
    """
    
    def __init__(self,params_path,data_path,n1trust,n2trust=0):
        super(MOTinterface,self).__init__()
        self.loading_curve = np.empty([0,2])
        self._t0 = np.float64(1.2)
        self._n1_trust_radius = np.float64(n1trust)
        self._n2_trust_radius = np.float64(n2trust)
        self._params_path = params_path
        self._data_path = data_path
        self._settings = configparser.ConfigParser()        
    
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
        
        coilscurrent = str(params[0])
        coolingfrequency = str(params[1])
        
        self._settings.read(self._params_path)
        
        self._settings['MOT']['coilscurrent'] = coilscurrent
        self._settings['MOT']['coolingfrequency'] = coolingfrequency
        
        with open(self._params_path, 'w') as settingsfile:
            self._settings.write(settingsfile)
        
        #at this point, the lab software has to somehow know that the
        #params.ini file has been updated and run the experiment. 
        #If there's anything that needs to be done on the M-LOOP side
        #for that to happen, this is where that would go.
        
        time.sleep(15)
        
        while not os.path.exists(self._data_path):
            time.sleep(5)
            print('Curve file not found, waiting...')
            pass
        
        self.loading_curve = cleaner.load_curve(self._data_path)
        self.loading_curve = cleaner.reset_origin(self._t0, self.loading_curve)
        os.remove(self._data_path)
        
        fitting = curvefit.loading_curve_fitting(self.loading_curve, self._n1_trust_radius, self._n2_trust_radius)
        fitting.fitting_routine_a()
        
        cost = fitting._g/fitting._R
        uncer = 1e-3
        bad = False
    
        cost_dict = {'cost' : cost, 'uncer' : uncer, 'bad' : bad}

        return cost_dict 