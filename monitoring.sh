#!/bin/bash

# malwares=("delay" "disorder" "freeze" "hop" "mimic" "noise" "normal" "normal_v2" "repeat" "spoof" "stop")
malwares=("normal" "delay" "stop")
for ware in ${malwares[@]}
	do
		if [[ $ware == "stop" ]]
		then
			#STOP MONITORING AND SET NORMAL EXECUTABLE
			echo "Stopping monitoring script..."
			pid=$(ps aux | grep get_system_trace | grep -v grep | head -n 1 | awk "{print $2}")
			# echo "Killing monitoring script. pid: " $pid
			# kill -9 $pid
			sleep 1
			echo "Restarting normal es_sensor executable"
			service electrosense-sensor-mqtt stop
			cp normal /usr/bin/es_sensor
			service electrosense-sensor-mqtt start
			echo "Done."
		else
			#SET MALICIOUS EXECUTABLE AND START MONITORING
			echo "Restarting es_sensor executable with new behavior for $ware"
			service electrosense-sensor-mqtt stop
			cp $ware /usr/bin/es_sensor
			service electrosense-sensor-mqtt start
			sleep 10
			echo "Starting monitoring script"
			./get_system_perf.sh $ware &
			# wait until get trace finished
			wait
			echo "Done."
			echo "================================================================="
			sleep 10
		fi
	done
echo "finished!!!"