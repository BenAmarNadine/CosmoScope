#!/bin/bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
rm -f $SCRIPTPATH/$1/env.sh
cp $SCRIPTPATH/env.sh $SCRIPTPATH/$1/env.sh
source "$SCRIPTPATH/$1/env.sh"

if [ ! $2 ]; then
  echo "Please input server index (0 ~ 3)"
  exit
fi

DATA_DIR=$SCRIPTPATH/$1

INDEX=`expr $2`

NODE_NAME="node"$((${INDEX}))

CURRENT_DATA_DIR=$SCRIPTPATH/$NODE_NAME
rm -rf $CURRENT_DATA_DIR
cp -r $DATA_DIR/$NODE_NAME $CURRENT_DATA_DIR

echo "NODE_NAME: ${NODE_NAME}"
echo "CURRENT_DATA_DIR: ${CURRENT_DATA_DIR}"

echo "RPC_PORT: ${RPC_PORTS[$INDEX]}"
echo "REAPCHAIN_PORT: ${REAPCHAIN_PORTS[$INDEX]}"

echo "cp $DATA_DIR/genesis.json $CURRENT_DATA_DIR/config/genesis.json"
cp $DATA_DIR/genesis.json $CURRENT_DATA_DIR/config/genesis.json



# Only enable API Server, the rest disable
sed -i.temp 's/enable = false/enable = true/g' $CURRENT_DATA_DIR/config/app.toml

# [explorer / wallet 변경사항]-------------------------------------------------------------------------------------------------------------------------- 

# EnableUnsafeCORS defines if CORS should be enabled (unsafe - use it at your own risk).
sed -i.temp 's/enabled-unsafe-cors = false/enabled-unsafe-cors = true/g' $CURRENT_DATA_DIR/config/app.toml

sed -i.temp 's/cors_allowed_origins = \[\]/cors_allowed_origins = \["\*"\]/g' $CURRENT_DATA_DIR/config/config.toml
sed -i.temp 's/cors_allowed_headers = \["Origin", "Accept", "Content-Type", "X-Requested-With", "X-Server-Time", \]/cors_allowed_headers = \["Origin", "Accept", "Content-Type", "X-Requested-With", "X-Server-Time", "Access-Control-Request-Method", "Access-Control-Request-Headers", "Authorization", \]/g' $CURRENT_DATA_DIR/config/config.toml

# 자신의 RCP 주소 설정
# echo "sed -i.temp "s/laddr = \"tcp:\/\/127.0.0.1:26657\"/laddr = \"tcp:\/\/${HOSTS[$INDEX]}:${RPC_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml"
# sed -i.temp "s/laddr = \"tcp:\/\/127.0.0.1:26657\"/laddr = \"tcp:\/\/${HOSTS[$INDEX]}:${RPC_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml
echo "sed -i.temp "s/laddr = \"tcp:\/\/127.0.0.1:26657\"/laddr = \"tcp:\/\/${PRIVATE_HOSTS[$INDEX]}:${RPC_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml"
sed -i.temp "s/laddr = \"tcp:\/\/127.0.0.1:26657\"/laddr = \"tcp:\/\/${PRIVATE_HOSTS[$INDEX]}:${RPC_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml
# sed -i.temp "s/laddr = \"tcp:\/\/127.0.0.1:26657\"/laddr = \"tcp:\/\/127.0.0.1:${RPC_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml

# REAPCHAIN 주소 설정
# sed -i.temp "s/laddr = \"tcp:\/\/0.0.0.0:26656\"/laddr = \"tcp:\/\/${HOSTS[$INDEX]}:${REAPCHAIN_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml 
sed -i.temp "s/laddr = \"tcp:\/\/0.0.0.0:26656\"/laddr = \"tcp:\/\/${PRIVATE_HOSTS[$INDEX]}:${REAPCHAIN_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml 
# sed -i.temp "s/laddr = \"tcp:\/\/0.0.0.0:26656\"/laddr = \"tcp:\/\/0.0.0.0:${REAPCHAIN_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml 

# 중복 IP 허용
sed -i.temp 's/allow_duplicate_ip = false/allow_duplicate_ip = true/g' $CURRENT_DATA_DIR/config/config.toml

# Mempool Size
sed -i.temp 's/size = 5000/size = 60000/g' $CURRENT_DATA_DIR/config/config.toml

# [Minimum Gas Prices (0.1*200,000 =20,000ureap)->0.02reap Fees]
sed -i.temp 's/minimum-gas-prices = \"\"/minimum-gas-prices = \"0stake\"/g' $CURRENT_DATA_DIR/config/app.toml

############################################################################################################################################
#[config create-empty-block 설정]
# sed -i "s/create_empty_blocks = true/create_empty_blocks = false/g" $CURRENT_DATA_DIR/config/config.toml
#sed -i "s/create_empty_blocks_interval = \"30s\"/create_empty_blocks_interval = \"2s\"/g" $CURRENT_DATA_DIR/config/config.toml

#[consensus config timeout 설정]
sed -i "s/timeout_propose = \"3s\"/timeout_propose = \"1s\"/g" $CURRENT_DATA_DIR/config/config.toml
sed -i "s/timeout_propose_delta = \"500ms\"/timeout_propose_delta = \"250ms\"/g" $CURRENT_DATA_DIR/config/config.toml

sed -i "s/timeout_prevote = \"1s\"/timeout_prevote = \"165ms\"/g" $CURRENT_DATA_DIR/config/config.toml
sed -i "s/timeout_prevote_delta = \"500ms\"/timeout_prevote_delta = \"250ms\"/g" $CURRENT_DATA_DIR/config/config.toml

sed -i "s/timeout_precommit = \"1s\"/timeout_precommit = \"165ms\"/g" $CURRENT_DATA_DIR/config/config.toml
sed -i "s/timeout_precommit_delta = \"500ms\"/timeout_precommit_delta = \"250ms\"/g" $CURRENT_DATA_DIR/config/config.toml

sed -i "s/timeout_commit = \"5s\"/timeout_commit = \"830ms\"/g" $CURRENT_DATA_DIR/config/config.toml
# log (indexed block events ~ received proposal) 사이에 timed out 시간


# sed -i 's/recheck = true/recheck = false/g' $CURRENT_DATA_DIR/config/config.toml
# sed -i 's/broadcast = true/broadcast = false/g' $CURRENT_DATA_DIR/config/config.toml


sed -i 's/max_open_connections = 900/max_open_connections = 1800/g' $CURRENT_DATA_DIR/config/config.toml
# sed -i 's/skip_timeout_commit = false/skip_timeout_commit = true/g' $CURRENT_DATA_DIR/config/config.toml


###########################################################################################################################################

# #[recv_rate 변경사항]
# sed -i 's/recv_rate = 5120000/recv_rate = 20480000/g' $CURRENT_DATA_DIR/config/config.toml
# sed -i 's/send_rate = 5120000/send_rate = 20480000/g' $CURRENT_DATA_DIR/config/config.toml

# #Cache Size
# sed -i 's/cache_size = 5000/cache_size = 10000/g' $CURRENT_DATA_DIR/config/config.toml

# #[p2p: flush 변경사항]
#sed -i.temp 's/flush_throttle_timeout = "100ms"/flush_throttle_timeout = "50ms"/g' $CURRENT_DATA_DIR/config/config.toml

# #[p2p: max_packet_msg_payload_size]
# sed -i.temp 's/max_packet_msg_payload_size = 1024/max_packet_msg_payload_size = 10240/g' $CURRENT_DATA_DIR/config/config.toml

# sed -i 's/skip_timeout_commit = false/skip_timeout_commit = true/g' $CURRENT_DATA_DIR/config/config.toml
# sed -i "s/timeout_broadcast_tx_commit = \"10s\"/timeout_broadcast_tx_commit = \"5s\"/g" $CURRENT_DATA_DIR/config/config.toml


# config.toml
# persistent_peers_max_dial_period
# sed -i "s/persistent_peers_max_dial_period = \"0s\"/persistent_peers_max_dial_period = \"10s\"/g" $CURRENT_DATA_DIR/config/config.toml
# pex
# sed -i 's/pex = true/pex= false/g' $CURRENT_DATA_DIR/config/config.toml
# dial_timeout
# sed -i "s/dial_timeout = \"3s\"/dial_timeout = \"10s\"/g" $CURRENT_DATA_DIR/config/config.toml


# genesis.json

# time_iota_ms 
# sed -i "s/\"time_iota_ms\": \"1000\"/\"time_iota_ms\": \"500\"/g" $CURRENT_DATA_DIR/config/genesis.json


sed -i 's/addr_book_strict = true/addr_book_strict = false/g' $CURRENT_DATA_DIR/config/config.toml
# max_bytes
sed -i "s/\"max_bytes\": \"22020096\"/\"max_bytes\": \"88080384\"/g" $CURRENT_DATA_DIR/config/genesis.json
# peer_gossip_sleep_duration
sed -i "s/peer_gossip_sleep_duration = \"100ms\"/peer_gossip_sleep_duration = \"1ms\"/g" $CURRENT_DATA_DIR/config/config.toml


# [app.toml port]
sed -i.temp 's/address = "0.0.0.0:9090"/address = "'$HOST:${GRPC_PORTS[$INDEX]}'"/g' $CURRENT_DATA_DIR/config/app.toml
sed -i.temp 's/address = "0.0.0.0:9091"/address = "'$HOST:${WEB_GRPC_PORTS[$INDEX]}'"/g' $CURRENT_DATA_DIR/config/app.toml
sed -i.temp 's/address = ":8080"/address = "':${ROSETTA_PORTS[$INDEX]}'"/g' $CURRENT_DATA_DIR/config/app.toml
sed -i.temp "s/address = \"tcp:\/\/0.0.0.0:1317\"/address = \"tcp:\/\/0.0.0.0:${API_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/app.toml

# [pprof port]
sed -i.temp "s/pprof_laddr = \"localhost:6060\"/pprof_laddr = \"localhost:${PPROF_PORTS[$INDEX]}\"/g" $CURRENT_DATA_DIR/config/config.toml

# [persistent peers]
echo "Update persistent_peers"
i=0
 while [ $i -lt $NODE_COUNT ]; do
  PERSISTENT_PEERS=""
   j=0
  while [ $j -lt $NODE_COUNT ]; do
      # PERSISTENT_PEERS=$PERSISTENT_PEERS${NODE_IDS[$j]}'@'${HOSTS[$j]}':'${REAPCHAIN_PORTS[$j]}','
      PERSISTENT_PEERS=$PERSISTENT_PEERS${NODE_IDS[$j]}'@'${PRIVATE_HOSTS[$j]}':'${REAPCHAIN_PORTS[$j]}','
      j=$((${j}+1))
  done  
  PERSISTENT_PEERS=${PERSISTENT_PEERS%,} #%, = ,를 제거하겠다는 의미
  echo "PERSISTENT_PEERS : "$PERSISTENT_PEERS
  sed -i "s/persistent_peers = \"\"/persistent_peers = \"$PERSISTENT_PEERS\"/g" $CURRENT_DATA_DIR/config/config.toml
  i=$((${i}+1))
done
rm -rf $CURRENT_DATA_DIR/config/app.toml.temp
rm -rf $CURRENT_DATA_DIR/config/config.toml.temp
rm -rf $CURRENT_DATA_DIR/config/genesis.json.temp 
