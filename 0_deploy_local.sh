#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

TEMP_PATH=~/sdk_temp
rm -rf $TEMP_PATH
SCRIPT_NAME=multi_sdk_script
echo "cp -r $SCRIPTPATH $TEMP_PATH"
cp -r $SCRIPTPATH $TEMP_PATH

i=0
while [ $i -lt $NODE_COUNT ]; do  
  echo "sudo rm -rf yhseo@${HOSTS[$i]}:~/$SCRIPT_NAME"
  sudo sshpass -p dcs5583!! sudo rm -rf yhseo@${HOSTS[$i]}:~/$SCRIPT_NAME

  i=$((${i}+1))
done

i=0
while [ $i -lt $NODE_COUNT ]; do  
  echo "sudo scp -r $TEMP_PATH yhseo@${HOSTS[$i]}:~/$SCRIPT_NAME"
  sudo sshpass -p dcs5583!! sudo scp -r $TEMP_PATH yhseo@${HOSTS[$i]}:~/$SCRIPT_NAME
  
  i=$((${i}+1))
done

rm -rf $TEMP_PATH