#!/bin/bash
for j in {0..20}
   do
      sed -i "185c feature = features[$j]" dnn_cpu.py
      sed -i "185s/^/    /" dnn_cpu.py
      python3 dnn_cpu.py
   done
