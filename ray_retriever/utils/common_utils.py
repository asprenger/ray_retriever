import os
import time
import itertools
from math import ceil

def partition(lst, n):
    size = ceil(len(lst) / n)
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def current_time_millis():
    obj = time.gmtime(0)
    epoch = time.asctime(obj)
    return round(time.time()*1000)

def load_file(file_path:str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def obfuscate_password(password:str, start_pos=0):
    if password is None:
        return None
    if len(password) < start_pos:
        return password
    if start_pos < 0 or start_pos > len(password):
        raise ValueError('Invalid start_pos')
    return password[0:start_pos] + '*' * (len(password) - start_pos)

def get_dir_size(dir_path:str):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size
