import shutil
import os
import optimize
import logging
from datetime import datetime as dt

if not os.path.exists('./testing/testing_logs'):
    os.mkdir('./testing/testing_logs')

stream_formatter = logging.Formatter('%(levelname)s %(message)s')
file_formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)

file_handler = logging.FileHandler('./testing/testing_logs/test_{}'.format(dt.now().strftime('%d-%m-%Y-%H-%M')))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

log = logging.getLogger(__name__)
log.addHandler(stream_handler)
log.addHandler(file_handler)
log.setLevel(logging.DEBUG)
        
def load_test_settings():
    log.info('Initializing optimizer test.')
    shutil.copyfile('./settings.ini','./settings_bak.ini')
    log.debug('Original settings backup saved.')
    shutil.copyfile('./testing/test_settings.ini','./settings.ini')
    log.debug('Test settings loaded.')
    shutil.copyfile('./testing/test_params.ini','./testing/params.ini')
    log.debug('Test parameters loaded.')
    
    
def run():
    log.info('Starting test.')
    controller_config_dict = optimize.controller_config_dict()   
    log.debug('Controller settings loaded.')
    settings_dict = optimize.settings_dict()
    log.debug('Optimizer settings loaded.')
    optimize.main(controller_config_dict,**settings_dict)

    
def restore():
    log.info('Test run completed. Restoring original settings.')
    shutil.copyfile('./settings_bak.ini','./settings.ini')
    log.debug('Original settings restored.')
    os.remove('./settings_bak.ini')
    log.debug('Original settings backup removed.')
    os.remove('./testing/params.ini')
    log.debug('Test parameters removed.')
    log.info('Test completed.')
    
    
if __name__ == '__main__':
    load_test_settings()
    run()
    restore()

