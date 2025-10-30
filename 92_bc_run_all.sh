#!/bin/bash
source "./env.sh"



i=0
while [ $i -lt $NODE_COUNT ]; do
#while [ $i -lt 3 ]; do
  echo "sudo ~/$SCRIPT_NAME/2_run.sh f $i"
  sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} "~/$SCRIPT_NAME/2_run.sh n $i"
  i=$((${i}+1))
done
