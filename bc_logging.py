import re
import subprocess
import time

initial_commands = [
    "./99_bc_kill_all.sh",
    "./90_bc_deploy.sh",
    "./91_bc_init_all.sh",
    "./92_bc_run_all.sh"
]

for cmd in initial_commands:
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)

time.sleep(2)

# -------------------------------------------------------------------
# Start continuous logging
# -------------------------------------------------------------------
logging_command = "./97_bc_loggingRealTime.sh 0"
print(f"Starting logging command: {logging_command}")
process = subprocess.Popen(logging_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

log_filename = "captured_bc_logs.txt"
ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
commit_regex = re.compile(r"(committed state).*num_txs=(\d+)")

try:
    with open(log_filename, "w") as logfile:
        while True:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)  # Avoid high CPU usage
                continue

            clean_line = ansi_escape.sub('', line)  # Remove ANSI escape sequences
            logfile.write(clean_line)
            logfile.flush()
            print(line.rstrip())

            commit_match = commit_regex.search(clean_line)
            if commit_match:
                txs = int(commit_match.group(2))
                if txs > 0:
                    print(f"DEBUG: New transactions detected: {txs}")
                else:
                    print("DEBUG: Transaction count is zero. Waiting for new transactions.")

except KeyboardInterrupt:
    print("\nInterrupted. Killing all blockchain processes...")
    
finally:
    # Kill all BC processes on interrupt or error
    kill_cmd = "./99_bc_kill_all.sh"
    print(f"Running cleanup command: {kill_cmd}")
    subprocess.run(kill_cmd, shell=True)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    print("Cleanup completed. Exiting.")

