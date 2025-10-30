#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

i=0
while [ $i -lt $NODE_COUNT ]; do
  echo "sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} killall -9 $PROCESSNAME"
  sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} "killall -9 $PROCESSNAME"

  i=$((${i}+1))
done