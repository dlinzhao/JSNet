import os
import sys
import logging


def get_logger(name, save_dir=None, filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(filename)s %(lineno)d %(levelname)s %(message)s')
    shandler = logging.StreamHandler(stream=sys.stdout)
    shandler.setLevel(logging.INFO)
    shandler.setFormatter(formatter)

    logger.addHandler(shandler)
    if save_dir:
        fhandler = logging.FileHandler(os.path.join(save_dir, filename))
        fhandler.setLevel(logging.INFO)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

    return logger
