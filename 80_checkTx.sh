#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

if [ $# -ne 1 ] ; then
  echo "./80_check_tx.sh [TX_ID]"
  echo "=> ex) ./80_check_tx.sh 68FCDFDD00826DE484096DDE2F03DB87DDA238203624308E26440D800C1B9399"
  exit
fi

BINARY=$SCRIPTPATH/bin/linux/middlewared
NODE_URL=http://${HOSTS[1]}:${RPC_PORTS[1]}
TX_ID=$1

echo "===================================== Check Tx ===================================== "
$BINARY query tx $TX_ID --node $NODE_URL