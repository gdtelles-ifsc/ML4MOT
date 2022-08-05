import os
import configparser
import logging
from sys import exit
from pathlib import Path
import mloop.controllers as mlc
import mloop.visualizations as mlv
import cleaner
import curvefit
import interface
import warnings

def main(controller_config_dict,params_path,data_path,controller_type, gpu_info=True, ignore_deprecation_warnings=False):
    """Creates an instance of the MOTinterface class, then
    a controller with that interface and a series of 
    configurations (that can be found in the M-LOOP 
    documentation). Lastly, the optimize() method is called
    to start the process."""
    
    if ignore_deprecation_warnings:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
    
    interf = interface.MOTinterface(params_path,data_path,0.6)
    controller = mlc.create_controller(interf, controller_type,**controller_config_dict)
   
    if not gpu_info:
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
    
    controller.optimize()
    
    #mlv.show_all_default_visualizations(controller)
    
def settings_dict():
    settings = configparser.ConfigParser()
    settings_dict = {}

    if not os.path.exists('./settings.ini'):
        exit('settings.ini not found, aborting...')    

    settings.read('./settings.ini')
    settings_dict['params_path'] = Path(settings['PATH']['params_path'])
    settings_dict['data_path'] = Path(settings['PATH']['data_path'])
    settings_dict['controller_type'] = settings['CONTROLLER']['controller_type']
    settings_dict['gpu_info'] = settings['CONTROLLER'].getboolean('gpu_info')
    settings_dict['ignore_deprecation_warnings'] = settings['CONTROLLER'].getboolean('ignore_deprecation_warnings') 

    return settings_dict

def controller_config_dict():
      return {'num_params'     : 2,
            'min_boundary'   : [0.0, -4.0],
            'max_boundary'   : [0.2, 4.0],
            'first_params'   : [0.09, 1.15],
            'max_num_runs'   : 10,
            'cost_has_noise' : True,
            'no_delay'       : False}


if __name__ == '__main__':
    controller_config_dict = controller_config_dict()   
    settings_dict = settings_dict()
    main(controller_config_dict,**settings_dict)