#!/bin/bash

current=`date "+%Y-%m-%d_%H:%M:%S"`;
nohup ./monitoring.sh >/data/attack_setup/hup_${current}.log 2>&1 &