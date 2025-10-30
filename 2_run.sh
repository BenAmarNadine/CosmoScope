#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/test/env.sh"

if [ ! $2 ]; then
  echo "Please input server index ( 0 ~ 3 )"
  exit
fi

RUN_TYPE=$1
INDEX=`expr $2`
BINARY=$SCRIPTPATH/bin/$PROCESSNAME
#DATA_DIR=$1

NODE_NAME="node"$((${INDEX}))

LOG_NAME="${NODE_NAME}_log"
CURRENT_DATA_DIR=$SCRIPTPATH/$NODE_NAME
 
if [ $RUN_TYPE = "f" ]; then
  echo "$BINARY start --home $CURRENT_DATA_DIR "
  $BINARY start --home $CURRENT_DATA_DIR 
else
  echo "$BINARY start --home $CURRENT_DATA_DIR &> $CURRENT_DATA_DIR/$LOG_NAME.log &"
  $BINARY start --home $CURRENT_DATA_DIR &> $CURRENT_DATA_DIR/$LOG_NAME.log &

  # $BINARY start --log_level "debug" --home $CURRENT_DATA_DIR &> $CURRENT_DATA_DIR/$LOG_NAME.log & 
fi
