import shutil
import os
import time
import logging
from datetime import datetime as dt

stream_formatter = logging.Formatter('%(levelname)s %(message)s')
file_formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)

file_handler = logging.FileHandler('./testing/testing_logs/copydata_{}'.format(dt.now().strftime('%d-%m-%Y-%H-%M')))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

log = logging.getLogger(__name__)
log.addHandler(stream_handler)
log.addHandler(file_handler)
log.setLevel(logging.DEBUG)
        

def main():
    log.info('Copydata started, waiting.')
    time.sleep(30)
    log.debug('Copydata initial waiting period completed.')
    
    while os.path.exists('./settings_bak.ini'):
        log.debug('Copydata running loop, waiting.')
        time.sleep(10)
        log.debug('Copydata completed loop.')
        if not os.path.exists('./testing/MOTpd.csv'):
            log.debug('Copydata did not find MOTpd.')
            shutil.copyfile('./testing/test_curve1.csv','./testing/MOTpd.csv')
            log.info('Test MOTpd copied.')
            
    log.debug('Copydata has completed the loop.')
    if os.path.exists('./testing/MOTpd.csv'):
        log.debug('Copydata has found MOTpd')
        os.remove('./testing/MOTpd.csv')
        log.info('Test MOTpd removed.')
                
if __name__ == '__main__':
    main()
    log.info('Copydata done.')
