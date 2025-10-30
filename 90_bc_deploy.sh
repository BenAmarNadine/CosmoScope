#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

TEMP_PATH=~/sdk_temp
# rm -rf $TEMP_PATH
echo "cp -r $SCRIPTPATH $TEMP_PATH"
cp -r $SCRIPTPATH $TEMP_PATH

i=0
while [ $i -lt $NODE_COUNT ]; do
#while [ $i -lt 3 ]; do
  echo "rm -rf ~/$SCRIPT_NAME"
  sudo ssh -i ./bc-net.pem -o StrictHostKeyChecking=no $USER@${HOSTS[$i]} "rm -rf ~/$SCRIPT_NAME"

  echo "sudo scp -i ./main-net.pem -o StrictHostKeyChecking=no -r $TEMP_PATH $USER@${HOSTS[$i]}:~/$SCRIPT_NAME"
  sudo scp -i ./bc-net.pem -o StrictHostKeyChecking=no -r $TEMP_PATH $USER@${HOSTS[$i]}:~/$SCRIPT_NAME

  i=$((${i}+$PROC_PER_IP))
done

echo "#Done nodes deploy ########"

rm -rf $TEMP_PATH
