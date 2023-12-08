import logging

def get_logger():
    if "ray.serve" in logging.Logger.manager.loggerDict.keys():
        logger = logging.getLogger("ray.serve")
    else:
        logger = logging.getLogger()
    return logger