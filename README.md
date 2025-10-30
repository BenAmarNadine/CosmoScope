# Reapchain Local Environment Scripts

### Local TestNet

#### 1. Setup binary
1. Please check replacement binary for testing
2. if so, copy it you want to test to appropriate location.
* binary is two type: mac and windows
<pre>
<code>
cp reapchaind ~reapchain-scripts/mercury_2022-2/local/bin 
</code>
</pre> 
3. if you have mac-os enviroment, if not, coreutil and gnu-sed must be installed
<pre>
<code>
brew install gnu-sed
brew install coreutils
</code>
</pre> 

#### 2. Auto Script
Run ./launch_local_testnet.sh [NUMBER_OF_STANDING_NODES] [NUMBER_OF_STEERING_CANDIDATE_NODES] [DIRECTORY]

You can launch a complete local testnet with this command. 

The [DIRECTORY] argument will setup all of the files and resources needed to run the testnet in the specified directory.

#### 3. Manual Method
./10_setup [NUMBER_OF_STANDING_NODES] [NUMBER_OF_STEERING_CANDIDATE_NODES] [DIRECTORY]

./11_init_all.sh [DIRECTORY]

./12_run_all.sh [DIRECTORY]

#### 4. Additional Scripts
You can stop all nodes with the ./21_stop_all.sh

You can check for running nodes with the ./20_check_running_process.sh 

#### 5. for launching with Single Node.  
Run ./single-node.sh

#### 6. for launching with normal Node.
1. Please check running test-net.
2. Run ./normal-node.sh [VALIDATOR ADDRESS@ip:port][[VALIDATOR DIRECTORY/config]
3. Please refer following code
<pre>
<code>
./normal_node.sh 4e0537b23178b4696b103d1782ac30e0db4dd4f2@172.25.155.173:37000 chain-data/node0/config 
</code>
</pre> 

#### 7. for creating validator.
1. Please check running test-net.
2. Please check running normal-node.
3. Run ./create-validator.sh [VALIDATOR DIRECTORY] [NORMAL NODE DIRECTORY] [VALIDATOR PORT NUMBER]
4. Please refer following code
<pre>
<code>
./create-validator.sh ./xx/node0 ./normal-node 37100 
</code>
</pre> 
5. Retry tx staking create-validator command() without "--gas auto" flag
<pre>
<code>
./bin/mac/reapchaind tx staking create-validator --amount 10000000000000000000areap --node-id 5b8fa706ff4d3fd91e80a69a1569c3cb743d6094 --moniker added-node --keyring-dir ./normal-node --from reap1xda398utcun7lragltuecsjgvh3t3wcejnyu74 --home ./normal-node --chain-id mercury_2022-2 --pubkey '{"@type":"/cosmos.crypto.ed25519.PubKey","key":"zoAymczLJr7NCoJcCXyuANnyRjetsWa+grxON02YttE="}' --commission-rate 0.10 --commission-max-rate 0.20 --commission-max-change-rate 0.01 --node http://192.168.0.47:37100 --min-self-delegation=10000000000000000000 --keyring-backend test --type steering --yes
</code>
</pre> 