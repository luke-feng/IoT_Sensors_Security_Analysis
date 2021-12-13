#!/bin/bash
for i in {1..4}
do
   for j in {1..21}
   do
      sed -i "90c feature = features[$j]" cpu.py
      sed -i "90s/^/    /" cpu.py
      sed -i "91c clsname = clss[$i]" cpu.py
      sed -i "91s/^/    /" cpu.py
      python3 cpu.py
   done
done