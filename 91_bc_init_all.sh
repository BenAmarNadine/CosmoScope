#!/bin/bash
source "./env.sh"

./99_bc_kill_all.sh

i=0
while [ $i -lt $NODE_COUNT ]; do
#while [ $i -lt 3 ]; do
  echo "sudo ~/$SCRIPT_NAME/1_init.sh test $i"
  sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} "~/$SCRIPT_NAME/1_init.sh test $i"
  i=$((${i}+1))
done
