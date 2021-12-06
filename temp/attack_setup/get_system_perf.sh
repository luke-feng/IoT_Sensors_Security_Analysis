#!/bin/bash

total_loop=360;
step=0;
time_window=60;
# the total monitoring time is total_loop * time_window
while(($step<$total_loop));
	do
		# get the pid
		mount -o remount rw /
		pid=$(ps aux | grep es_sensor | grep -v sudo | grep -v grep | awk '{print $2}');
		while [[ $pid == "" ]];
			do 
				# try to get the pid again
				sleep 10
				service electrosense-sensor-mqtt stop
				echo "Restarting behavior for $1"
				cp $1 /usr/bin/es_sensor
				service electrosense-sensor-mqtt start
				sleep 10 
				pid=$(ps aux | grep es_sensor | grep -v sudo | grep -v grep | awk '{print $2}');
			done
		echo "start to perf system calls for $1 in loop $step, pid is $pid"
		current=`date "+%Y-%m-%d_%H:%M:%S"`;
		path="$1_${pid}_${step}_${current}";
		# start perf and save results, without nanosleep
		timeout -s 1 ${time_window} perf trace -o /data/attack_setup/dataset/${path}.txt -e !nanosleep -T -p ${pid};
		python preprocessing_pref.py  /data/attack_setup/dataset/${path}.txt /data/attack_setup/dataset/${path}.csv
		rm -rf /data/attack_setup/dataset/${path}.txt
		echo "step ${step} finish";
		((step=$step+1));
	done;
echo "get system calls for $1 in ${total_loop} loops finished";