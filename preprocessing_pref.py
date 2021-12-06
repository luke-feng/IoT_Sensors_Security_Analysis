import re
import sys

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

def main(argv):
   inputfile = argv[0]
   outputfile = argv[1]
   with open(inputfile, 'r') as f, open(outputfile, 'w') as outp:
        for line in f:
            res = get_systemcall_name_perf(line)
            if res != None:
                [pid, timestamp, syscall, time_cost] = res
                outp.write('{},{},{},{}\n'.format(pid, timestamp, syscall, time_cost))
        f.close()
        outp.close()

if __name__ == "__main__":
    main(sys.argv[1:])