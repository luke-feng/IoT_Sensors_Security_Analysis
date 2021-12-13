#!/bin/bash
   for j in {2..9}
   do
      sed -i "81c ftname = features[$j]" pca_cpu.py
      sed -i "81s/^/    /" pca_cpu.py
      python3 pca_cpu.py
   done