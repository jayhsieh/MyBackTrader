import logging
import os
from logging.handlers import TimedRotatingFileHandler


def log_setup(logger_str):
    log_folder = "./log"
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    logger = logging.getLogger(logger_str)
    logger.setLevel(logging.DEBUG)

    time_format = "%(asctime)s.%(msecs)03d %(levelname)s %(message)s"
    formatter = logging.Formatter(fmt=time_format,
                                  datefmt="%H:%M:%S")
    fh = TimedRotatingFileHandler(f'{log_folder}/{logger_str}.log',
                                  when='midnight', interval=1)
    fh.suffix = "%Y%m%d"
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
