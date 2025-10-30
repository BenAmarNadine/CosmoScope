#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source "$SCRIPTPATH/env.sh"

i=0
tx="Co8BCowBChwvY29zbW9zLmJhbmsudjFiZXRhMS5Nc2dTZW5kEmwKLW1zc29uZzE5azZjenN0dnFkNzIyc2U3a2g4NzVrdDJuZHRodXMwMDBjNG1uMBItbXNzb25nMTlrNmN6c3R2cWQ3MjJzZTdraDg3NWt0Mm5kdGh1czAwMGM0bW4wGgwKBXN0YWtlEgMxMDASVgpOCkYKHy9jb3Ntb3MuY3J5cHRvLnNlY3AyNTZrMS5QdWJLZXkSIwohArjYwEr+O8P1VrG3X2ZuxYyKvAxYImPyaeAsvTqIsVT0EgQKAggBEgQQwJoMGkCsPC4VB+KPEmUt5hfyFI9swKZi3HY2mBAoGBEUfeuqNjNERdpwQ2E4vQ2ETGZhl2L+iuxVfiL+p4t9P7/yfN9C"
# echo "==================================Send tx====================================="
curl -X POST -H "Content-Type: application/json" -d '{"tx_bytes":"'$tx'","mode":"BROADCAST_MODE_SYNC"}' "http://${HOSTS[$i]}:${API_PORTS[$i]}/cosmos/tx/v1beta1/txs"

