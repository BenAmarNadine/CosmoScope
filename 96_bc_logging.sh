#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

i=0
echo "sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} "tail -10000f $SCRIPT_NAME/node0/node0_log.log | grep -e 'num_valid_txs' -e 'committed state'""
sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} "tail -10000f $SCRIPT_NAME/node0/node0_log.log | grep -e 'num_valid_txs' -e 'committed state'"

