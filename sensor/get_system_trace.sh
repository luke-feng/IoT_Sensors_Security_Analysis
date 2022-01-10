#!/bin/bash

total_loop=36;
step=0;
time_window=600;
# the total monitoring time is total_loop * time_window
while(($step<$total_loop));
	do
		# get the pid
		pid=$(ps aux | grep es_sensor | grep -v sudo | grep -v grep | awk '{print $2}');
		# pid=$(ps aux | grep rssh | awk '/electro+/{print $2}');
		while [[ $pid == "" ]];
			do 
				# try to get the pid again
				echo "Restarting behavior for $1"
				cp $1 /usr/bin/es_sensor
				service electrosense-sensor-mqtt start
				sleep 10 
				pid=$(ps aux | grep es_sensor | grep -v sudo | grep -v grep | awk '{print $2}');
				# pid=$(ps aux | grep rssh | awk '/electro+/{print $2}');
			done
		echo "start to strace system calls for $1 in loop $step"
		current=`date "+%Y-%m-%d_%H:%M:%S"`;
		path="$1_${pid}_${step}_${current}";
		# start strace and save results
		timeout -s 2 ${time_window} strace -e trace=all -o /data/attack_setup/dataset/${path}.txt -f -F -t -T -p ${pid};
		echo "step ${step} finish";
		((step=$step+1));
	done;
echo "get system calls for $1 in ${total_loop} loops finished";

