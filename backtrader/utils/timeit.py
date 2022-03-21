#!/usr/bin/env python
import time


def measuretime(func):
    """
    Measuring execution time
    :param func:
    :return:
    """
    def wrapper():
        starttime = time.perf_counter()
        func()
        endtime = time.perf_counter()
        print(f"{func.__name__} ===> Time needed: {endtime - starttime} seconds")

    return wrapper
