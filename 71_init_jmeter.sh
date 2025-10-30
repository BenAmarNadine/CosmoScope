#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

#i=$1
i=0

echo "scp jmx file"
sudo ssh -i main-net.pem ubuntu@${JMETERS[i]} "rm -rf ~/apache-jmeter-5.4.1/*.jmx"
# ssh -i main-net.pem ubuntu@${JMETERS[i]} "rm -rf ~/apache-jmeter-5.4.1/middleware_tps.jmx"
sudo scp -i main-net.pem bin/*.jmx ubuntu@${JMETERS[i]}:~/apache-jmeter-5.4.1/

echo "scp csv file"
sudo ssh -i main-net.pem ubuntu@${JMETERS[i]} "rm -rf ~/apache-jmeter-5.4.1/*.csv"
# ssh -i main-net.pem ubuntu@${JMETERS[i]} "rm -rf ~/apache-jmeter-5.4.1/txs25000.csv" 
sudo scp -i main-net.pem bin/*.csv ubuntu@${JMETERS[i]}:~/apache-jmeter-5.4.1/
