#!/bin/bash

string="1719127.961 ( 0.008 ms): es_sensor/971 ioctl(arg0: 8, arg1: 1, arg2: 1977539428, arg3: 0, arg4: 1914702576, arg5: 29048760) = 0"
# modify quantifier to adjust the number of figures
pattern='s/([[:digit:]]{1,}.[[:digit:]]{1,}) \( ([[:digit:]]{1,}.[[:digit:]]{1,}) ms\): es_sensor\/([[:digit:]]{1,}) ([A-Za-z0-9_]*)\(arg0.*/\1, \2, \3, \4/p'
echo "$string" | sed -rn "$pattern"


freeze_20257_132_2021-11-20_09:21:31.txt

sed -rn "$pattern" normal_936_0_2021-11-19_12:47:12.txt > t1.txt