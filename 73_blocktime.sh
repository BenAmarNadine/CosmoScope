#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

#./7_aws_blocktime.sh [height]
# curl http://${HOSTS[0]}:27000/block?height=$1

curl http://${HOSTS[0]}:27000/block?height=$1 | grep txs

curl http://${HOSTS[0]}:27000/block?height=$1 | grep time