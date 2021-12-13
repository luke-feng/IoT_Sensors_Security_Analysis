import re
import sys
import time
import os
from numpy.core.fromnumeric import mean
import multiprocessing as mp
import numpy as np
import psutil

def get_systemcall_name_perf(line):
    l = re.split(r' |\( |\)', line)
    l = list(filter(lambda a: a != '', l))
    # print(l)
    if len(l) < 6:
        return None
    timestamp = l[0]
    time_cost = l[1]
    pid = l[4]
    if l[5] == '...':
        if timestamp == '0.000':
            return None
        else:
            syscall = l[7].split('(')[0]
    else:
        syscall = l[5].split('(')[0]
    # print(timestamp,time_cost, pid,  syscall)
    return [pid, timestamp, syscall, time_cost]

def main():
    currentPath = os.getcwd()
    # print(currentPath)
    raw_datapath = currentPath+'/raw_data/'
    perd_datapath = currentPath+'/pred_data/'
    files = os.listdir(raw_datapath)
    # for f in files:
    f = files[0]
    if '.txt' in f:
        inputfile = raw_datapath + f
        outputfile = perd_datapath + f
        with open(inputfile, 'r') as f, open(outputfile, 'w') as outp:
            # t1 = time.time()
            for line in f:
                res = get_systemcall_name_perf(line)
                if res != None:
                    [pid, timestamp, syscall, time_cost] = res
                    outp.write('{},{},{},{}\n'.format(pid, timestamp, syscall, time_cost))
            f.close()
            outp.close()
                # t2 = time.time()
                # t = t2 - t1
                # print(t)

def test2(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    mem_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        mem_percents.append(p.memory_full_info().rss)
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents, mem_percents


if __name__ == "__main__":
    cpu_percents, mem_percents = test2(target=main)
    print(mean(cpu_percents))
    print(mean(mem_percents)) 