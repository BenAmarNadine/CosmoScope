import os
import re
import csv
import time
import json
import requests
import subprocess
import base64
import hashlib
import tempfile
import sys
from datetime import datetime
from collections import defaultdict

if len(sys.argv) != 2:
    print("Usage: python3 log_monitor.py <TOTAL_TXS_TO_MONITOR|mainnet>")
    sys.exit(1)

label = sys.argv[1]
if label.isdigit():
    label_str = f"{label}tx"
else:
    label_str = str(label)

os.makedirs("results", exist_ok=True)
CSV_FILE = f"results/bench_metrics_{label_str}.csv"
VP_CHANGE_FILE = f"results/vp_change_events_{label_str}.csv"
MEMPOOL_SIZE_FILE = f"results/mempool_size_{label_str}.csv"
TX_LATENCY_FILE = f"results/tx_latency_{label_str}.csv"
LOG_FILE = "captured_bc_logs.txt"

execution_regex = re.compile(
    r"executed block height=(?P<height>\d+).*"
    r"num_invalid_txs=(?P<invalid>\d+).*"
    r"num_valid_txs=(?P<valid>\d+).*"
    r"timestamp=(?P<ts>\d{2}:\d{2}:\d{2}\.\d{3})"
)
commit_regex = re.compile(r"committed state .*height=(\d+).*timestamp=(\d{2}:\d{2}:\d{2}\.\d{3})")
consensus_step_regex = re.compile(r'Consensus Step Duration duration_ms=(\d+)\s+height=(\d+).*round=(\d+)\s+step="([^"]+)"')
block_size_regex = re.compile(r"Block size info height=(\d+).*size_bytes=(\d+)")
mempool_size_regex = re.compile(
    r"\[Mempool\]\s+avg_wait_time=(\d+)\s+max_wait_time=(\d+)\s+size=(\d+)\s+timestamp=(\d{2}:\d{2}:\d{2}\.\d{3})"
)
tx_latency_regex = re.compile(
    r"\[TxLatency\]\s+block_height=(\d+)\s+commit_time=(\d{2}:\d{2}:\d{2}\.\d{3})\s+latency_ms=(\d+)\s+submission_time=(\d{2}:\d{2}:\d{2}\.\d{3})\s+tx_hash=([A-F0-9]+)"
)

def source_env(env_path="./env.sh"):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as temp_script:
        temp_script.write(f"""
            #!/bin/bash
            source {env_path}
            echo "GENESIS_DIR=$GENESIS_DIR"
            echo "NODE_ROOT_DIR=$NODE_ROOT_DIR"
            env
        """)
        temp_script_path = temp_script.name

    result = subprocess.run(["bash", temp_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.unlink(temp_script_path)

    if result.returncode != 0:
        print(f"[ERROR] Failed to source env.sh: {result.stderr}")
        return

    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split("=", 1)
            os.environ[key] = value

def load_rpc_url_from_env(env_path="./env.sh"):
    with open(env_path, "r") as f:
        content = f.read()
    host_match = re.search(r'HOSTS\[0\]="([^"]+)"', content)
    port_match = re.search(r'RPC_PORTS\[0\]="([^"]+)"', content)
    if host_match and port_match:
        return f"http://{host_match.group(1)}:{port_match.group(1)}"
    else:
        raise ValueError("Could not find HOSTS[0] or RPC_PORTS[0] in env.sh")

def pubkey_to_address(base64_key):
    key_bytes = base64.b64decode(base64_key)
    sha256_hash = hashlib.sha256(key_bytes).digest()
    return sha256_hash[:20].hex().upper()

def update_voting_power_map():
    POWER_REDUCTION = 1_000_000
    try:
        result = subprocess.run(
            ["cosmosTestd", "q", "staking", "validators", "--output", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        validators = data.get("validators", [])
        vp_map = {}
        # Always write validators.csv (latest snapshot)
        with open("validators.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["address", "voting_power"])
            for val in validators:
                base64_key = val["consensus_pubkey"]["key"]
                address = pubkey_to_address(base64_key)
                tokens = int(val["tokens"])
                voting_power = tokens // POWER_REDUCTION
                vp_map[address] = voting_power
                writer.writerow([address, voting_power])
        print(f"[INFO] Updated voting power map for {len(vp_map)} validators.")
        return vp_map
    except Exception as e:
        print(f"[ERROR] Failed to update voting power map: {e}")
        return {}

def get_block_proposer(rpc_url, height):
    try:
        resp = requests.get(f"{rpc_url}/block?height={height}").json()
        return resp["result"]["block"]["header"]["proposer_address"]
    except Exception:
        return None

def get_proposer_voting_power(rpc_url, height, vp_map):
    proposer = get_block_proposer(rpc_url, height)
    proposer_vp = vp_map.get(proposer, 0)
    return proposer, proposer_vp

def load_max_bytes():
    genesis_dir = os.environ.get("GENESIS_DIR")
    if not genesis_dir:
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    genesis_path = os.path.join(script_dir, "..", genesis_dir, "genesis.json")
    try:
        with open(genesis_path, "r") as f:
            data = json.load(f)
            return int(data["consensus_params"]["block"]["max_bytes"])
    except Exception:
        return None

def parse_timestamp(time_str):
    return datetime.strptime(time_str, "%H:%M:%S.%f")

def log_vp_changes(block_height, old_map, new_map):
    with open(VP_CHANGE_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        for addr, new_vp in new_map.items():
            old_vp = old_map.get(addr)
            if old_vp is not None and old_vp != new_vp:
                writer.writerow([block_height, addr, old_vp, new_vp])

# Initialize environment
source_env()
MAX_BYTES = load_max_bytes()
RPC_URL = load_rpc_url_from_env()
VP_MAP = update_voting_power_map()
prev_vp_map = VP_MAP.copy()
processed_blocks = set()
execution_times = {}
tx_counts = {}
commit_times = {}
block_sizes = {}
consensus_steps = defaultdict(list)

os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Block Height", "TX Count", "Execution Time", "Commit Time", "Block Duration (ms)",
        "Block Size (bytes)", "Block Space Used (%)",
        "Consensus Rounds", "Consensus Round Duration (ms)",
        "Propose → Prevote", "Prevote → Precommit", "Precommit → Commit",
        "Proposer Address", "Proposer VP", "TPS"
    ])

with open(MEMPOOL_SIZE_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Mempool Size", "Avg Mempool Wait Time", "Max Mempool Wait Time"])

# Write VP change header at the start
if not os.path.exists(VP_CHANGE_FILE):
    with open(VP_CHANGE_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Block Height", "Validator Address", "Old Voting Power", "New Voting Power"])

with open(TX_LATENCY_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Block Height", "Tx Hash", "Submission Time", "Commit Time", "Latency (ms)"])

def maybe_write_block(height):
    if (height in execution_times and height in commit_times and height in block_sizes and height not in processed_blocks):
        exec_time = execution_times[height]
        commit_time = commit_times[height]
        block_duration_ms = int((commit_time - exec_time).total_seconds() * 1000)
        size_bytes = block_sizes[height]
        size_pct = (size_bytes / MAX_BYTES) * 100 if MAX_BYTES else 0
        tx_count = tx_counts.get(height, 0)
        rounds_map = {}
        for rn, st, dur in consensus_steps.get(height, []):
            rounds_map.setdefault(rn, {})[st] = dur

        if rounds_map:
            steps = {}
            selected_round = -1
            for rn, stmap in sorted(rounds_map.items(), reverse=True):
                if all(k in stmap for k in ["Propose → Prevote", "Prevote → Precommit", "Precommit → Commit"]):
                    steps = stmap
                    selected_round = rn
                    break

            propose_to_prevote   = steps.get("Propose → Prevote", 0)
            prevote_to_precommit = steps.get("Prevote → Precommit", 0)
            precommit_to_commit  = steps.get("Precommit → Commit", 0)

            round_count = selected_round + 1 if selected_round != -1 else 0
            round_duration = propose_to_prevote + prevote_to_precommit + precommit_to_commit
        else:
            propose_to_prevote = prevote_to_precommit = precommit_to_commit = 0
            round_count = round_duration = 0

        proposer_address, proposer_vp = get_proposer_voting_power(RPC_URL, height, VP_MAP)
        tps = tx_count / (block_duration_ms / 1000.0) if block_duration_ms > 0 else 0

        row = [
            height, tx_count, exec_time.strftime("%H:%M:%S.%f")[:-3],
            commit_time.strftime("%H:%M:%S.%f")[:-3], block_duration_ms,
            size_bytes, round(size_pct, 2),
            round_count, round_duration,
            propose_to_prevote, prevote_to_precommit, precommit_to_commit,
            proposer_address, proposer_vp, tps
        ]
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        processed_blocks.add(height)
        consensus_steps.pop(height, None)
        print(f"[CSV] {row}")  # Print every CSV row to CMD

print("[INFO] Monitoring started...")

try:
    with open(LOG_FILE, "r") as log_file:
        log_file.seek(0, os.SEEK_END)
        total_committed = 0
        last_block_with_tx = 0
        all_rows = []
        while True:
            line = log_file.readline()
            if not line:
                time.sleep(0.1)
                continue

            if "Consensus Step Duration" in line:
                match = consensus_step_regex.search(line)
                if match:
                    duration  = int(match.group(1))
                    height    = int(match.group(2))
                    round_num = int(match.group(3))
                    step      = match.group(4)
                    consensus_steps[height].append((round_num, step, duration))

            elif "executed block" in line:
                match = execution_regex.search(line)
                if match:
                    height = int(match["height"])
                    tx_count = int(match["valid"])
                    exec_time = parse_timestamp(match["ts"])
                    execution_times[height] = exec_time
                    tx_counts[height] = tx_count
                    if tx_count > 0:
                        total_committed += tx_count
                        last_block_with_tx = height
                    # Update VP map and log changes every 5 blocks
                    if height % 5 == 0:
                        new_vp_map = update_voting_power_map()
                        log_vp_changes(height, prev_vp_map, new_vp_map)
                        prev_vp_map = new_vp_map
                        VP_MAP = new_vp_map
                        print(f"[INFO] Updated VP_MAP for block {height}")
                    maybe_write_block(height)

            elif "committed state" in line:
                match = commit_regex.search(line)
                if match:
                    height = int(match.group(1))
                    commit_time = parse_timestamp(match.group(2))
                    commit_times[height] = commit_time
                    maybe_write_block(height)

            elif "Block size info" in line:
                match = block_size_regex.search(line)
                if match:
                    height = int(match.group(1))
                    size_bytes = int(match.group(2))
                    block_sizes[height] = size_bytes
                    maybe_write_block(height)

            elif "[Mempool]" in line:
                match = mempool_size_regex.search(line)
                if match:
                    avg_wait_time = int(match.group(1))
                    max_wait_time = int(match.group(2))
                    size     = int(match.group(3))
                    ts       = parse_timestamp(match.group(4))
                    with open(MEMPOOL_SIZE_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S.%f"), size, avg_wait_time, max_wait_time])

            elif "[TxLatency]" in line:
                match = tx_latency_regex.search(line)
                if match:
                    block_height = int(match.group(1))
                    commit_time = match.group(2)
                    latency_ms = int(match.group(3))
                    submission_time = match.group(4)
                    tx_hash = match.group(5)
                    with open(TX_LATENCY_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([block_height, tx_hash, submission_time, commit_time, latency_ms])

        print(f"[INFO] Collected {total_committed} TXs (target: {label}), stopping monitoring.")

except KeyboardInterrupt:
    print("\n[INFO] Monitoring interrupted by user.")

# Remove trailing zero-tx blocks from the results CSV
block_rows = []
with open(CSV_FILE, "r") as f:
    reader = list(csv.reader(f))
    header = reader[0]
    for row in reader[1:]:
        if int(row[1]) > 0 or int(row[0]) <= last_block_with_tx:
            block_rows.append(row)

# Write filtered CSV (no trailing 0-tx blocks)
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in block_rows:
        writer.writerow(row)

# Calculate summary metrics on filtered data
all_tx = [int(row[1]) for row in block_rows]
all_durations = [float(row[4])/1000.0 for row in block_rows]
all_tps = [float(row[-1]) for row in block_rows]

total_tx = sum(all_tx)
total_time = sum(all_durations)
overall_tps = total_tx / total_time if total_time > 0 else 0

print(f"\n[SUMMARY] Total TXs: {total_tx}, Total time: {total_time:.2f}s, Achieved TPS: {overall_tps:.2f}")

# Append summary row to CSV
with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["SUMMARY", total_tx, "", "", f"{total_time:.2f}s", "", "", "", "", "", "", "", "", "", f"{overall_tps:.2f}"])

