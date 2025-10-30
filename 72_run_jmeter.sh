#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

#i=$1
i=0
echo "jmeter $i"
#sudo ssh -i main-net.pem ubuntu@${JMETERS[$1]} "rm broadcast_time"
#250b

sudo ssh -i main-net.pem ubuntu@${JMETERS[$i]} "apache-jmeter-5.4.1/bin/jmeter -n -t apache-jmeter-5.4.1/cosmos_tps.jmx"

