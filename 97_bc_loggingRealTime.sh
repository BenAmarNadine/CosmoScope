#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

if [ $# -ne 1 ] ; then
	echo "Error! Insert the argument (node number)!"
	exit
fi


i=$1
echo "sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} tail -f $SCRIPT_NAME/node${i}/node${i}_log.log"
sudo ssh -i ./bc-net.pem $USER@${HOSTS[$i]} tail -f $SCRIPT_NAME/node${i}/node${i}_log.log
