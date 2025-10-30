import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

plt.style.use("science")
mpl.rcParams["text.usetex"] = False
sns.set_context("talk")

figtree_font_path = os.path.expanduser("~/.local/share/fonts/Figtree.ttf")
if os.path.exists(figtree_font_path):
    font_manager.fontManager.addfont(figtree_font_path)
    font_prop = font_manager.FontProperties(fname=figtree_font_path)
    mpl.rcParams["font.family"] = font_prop.get_name()
    print(f"Using font: {font_prop.get_name()}")
else:
    print("Figtree font not found. Using default font.")

if len(sys.argv) < 2:
    print("Usage: python monitoring_plotter.py <LOAD> (e.g., 5000 for bench_metrics_5000tx.csv)")
    sys.exit(1)
LOAD = sys.argv[1]

# --- PLOTS DIRECTORY ---
plots_dir = f"plots_{LOAD}tx"
os.makedirs(plots_dir, exist_ok=True)

# File locations
bench_file = f"results/bench_metrics_{LOAD}tx.csv"
mempool_file = f"results/mempool_size_{LOAD}tx.csv"
validators_file = "validators.csv"

if not os.path.exists(bench_file):
    raise FileNotFoundError(f"Could not find {bench_file} in ./results/")
if not os.path.exists(mempool_file):
    raise FileNotFoundError(f"Could not find {mempool_file} in ./results/")
if not os.path.exists(validators_file):
    raise FileNotFoundError(f"Could not find {validators_file} in current directory.")

df = pd.read_csv(bench_file)
mempool_df = pd.read_csv(mempool_file)
validators_df = pd.read_csv(validators_file)

# Only keep blocks up to the last with TXs
last_nonzero_block = df[df["TX Count"] > 0]["Block Height"].max()
df_active = df[df["Block Height"] <= last_nonzero_block].copy()

validator_addresses = set(validators_df["address"])
print(f"[INFO] Total validators from file: {len(validator_addresses)}")

for col in ["Execution Time", "Commit Time"]:
    if col in df_active.columns:
        df_active[col] = pd.to_datetime(df_active[col], format="%H:%M:%S.%f", errors='coerce')

def safe_numeric(col):
    if col in df_active.columns:
        df_active[col] = pd.to_numeric(df_active[col], errors="coerce")

safe_numeric("Block Duration (ms)")
safe_numeric("Consensus Rounds")
safe_numeric("Proposer VP")

if df_active["Block Duration (ms)"].dtype not in [float, int]:
    print("[WARN] Non-numeric Block Duration (ms) values detected, auto-converting...")

df_active["Proposer Label"] = df_active["Proposer Address"].apply(lambda addr: addr[:6] + "..." + addr[-4:] if isinstance(addr, str) else addr)
if "Proposer VP" in df_active.columns:
    try:
        vp_bins = pd.qcut(df_active["Proposer VP"], q=6, duplicates="drop")
        df_active["VP Bin"] = vp_bins.apply(lambda x: f"{int(x.left)}–{int(x.right)}")
    except Exception:
        df_active["VP Bin"] = df_active["Proposer VP"]

if "Timestamp" in mempool_df.columns:
    mempool_df["Timestamp"] = pd.to_datetime(mempool_df["Timestamp"])

def find_first_zero_sustained(df, window=3):
    sizes = df["Mempool Size"].values
    for i in range(len(sizes) - window + 1):
        if all(s == 0 for s in sizes[i:i+window]):
            return df["Timestamp"].iloc[i]
    return df["Timestamp"].iloc[-1]  # fallback

cutoff_time = find_first_zero_sustained(mempool_df, window=3)
mempool_df_active = mempool_df[mempool_df["Timestamp"] < cutoff_time].copy()

def finalize_plot(xlabel, ylabel, title, filename, legend_loc=None):
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=13)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(visible=True, axis='y', linestyle='--', alpha=0.3)
    plt.tick_params(direction='in', length=4, width=1)
    if legend_loc and plt.gca().get_legend():
        plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{filename}.pdf")
    plt.savefig(f"{plots_dir}/{filename}.png")
    plt.close()

# Plot 1
plt.figure(figsize=(8, 5))
sns.histplot(df_active["Proposer VP"], bins=15, kde=False)
finalize_plot("Proposer Voting Power", "Block Proposals", "Proposer Voting Power Distribution", "plot1_vp_distribution")

# Plot 2
plt.figure(figsize=(8, 5))
q98 = df_active["Block Duration (ms)"].quantile(0.98)
normal = df_active[df_active["Block Duration (ms)"] <= q98]
outliers = df_active[df_active["Block Duration (ms)"] > q98]
sns.scatterplot(data=normal, x="Proposer VP", y="Block Duration (ms)", s=40)
sns.scatterplot(data=outliers, x="Proposer VP", y="Block Duration (ms)", s=40, color="red")
plt.yscale("log")
finalize_plot("Proposer Voting Power", "Block Duration (ms, log scale)", "Block Duration vs Proposer Voting Power", "plot2_duration_vs_vp")

# Plot 3
if "Consensus Rounds" in df_active.columns:
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(
        data=df_active,
        x="VP Bin",
        y="Consensus Rounds",
        inner="point",
        scale="width"  # Use density_norm='width' with newer seaborn
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45)
    finalize_plot(
        "Proposer VP (binned)",
        "Consensus Rounds",
        "Consensus Rounds vs Proposer Voting Power",
        "plot3_rounds_vs_vp"
    )

# Plot 4
plt.figure(figsize=(10, 5))
q995 = mempool_df_active["Mempool Size"].quantile(0.99)
mempool_outliers = mempool_df_active[mempool_df["Mempool Size"] > q995].sort_values("Mempool Size", ascending=False).head(5)
plt.plot(mempool_df_active["Timestamp"], mempool_df_active["Mempool Size"], color="tab:blue", linewidth=1.5)
plt.fill_between(mempool_df_active["Timestamp"], mempool_df_active["Mempool Size"], color="tab:blue", alpha=0.3)
y_max = mempool_df_active["Mempool Size"].max()
max_row = mempool_df_active[mempool_df["Mempool Size"] == y_max].iloc[0]
plt.scatter([max_row["Timestamp"]], [max_row["Mempool Size"]], color="red", s=30, zorder=5)
x_max = max_row["Timestamp"]
side = "left" if x_max > mempool_df_active["Timestamp"].median() else "right"
x_offset = pd.Timedelta(seconds=-3 if side == "left" else 3)
ha = "right" if side == "left" else "left"
label_y = y_max + (0.04 * y_max)
plt.axvline(x=x_max, color="black", linestyle="--", linewidth=1, alpha=0.4)
plt.annotate(f"{int(y_max)} TXs\n{x_max.strftime('%H:%M:%S')}",
        xy=(x_max, y_max), xytext=(x_max + x_offset, label_y), textcoords="data",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="black"), ha=ha, fontsize=9)
plt.ylim(bottom=0, top=y_max * 1.25)
if not mempool_outliers.empty:
    y_max = mempool_outliers["Mempool Size"].max()
    plt.ylim(top=y_max * 1.15)
else:
    plt.ylim(top=mempool_df_active["Mempool Size"].max() * 1.05)
finalize_plot("Mempool Timestamp", "Mempool Size (TXs)", "Mempool Size Over Time", "plot4_mempool_over_time")

# Plot 5
plt.figure(figsize=(10, 5))
plt.plot(
    df_active["Block Height"], 
    df_active["Block Duration (ms)"], 
    marker="o", markersize=3, linewidth=1, color="tab:blue"
)
plt.scatter(
    outliers["Block Height"], 
    outliers["Block Duration (ms)"], 
    color="red", s=40, zorder=5
)
finalize_plot("Block Height", "Block Duration (ms)", "Block Duration Over Time", "plot5_duration_over_time")

# Plot 6
proposer_vp_map = dict(zip(validators_df["address"], validators_df.get("voting_power", [0]*len(validators_df))))
proposer_counts_raw = df_active["Proposer Address"].value_counts().to_dict()
all_proposers = list(validator_addresses)
proposer_vp = [proposer_vp_map.get(addr, 0) for addr in all_proposers]
proposer_blocks = [proposer_counts_raw.get(addr, 0) for addr in all_proposers]
proposer_df = pd.DataFrame({
    "address": all_proposers,
    "voting_power": proposer_vp,
    "blocks_proposed": proposer_blocks
})
proposer_df = proposer_df.sort_values(by="voting_power", ascending=False)
colors = ["#d7263d"]*5 + ["#2584a5"]*(len(proposer_df)-5)

plt.figure(figsize=(18, 4.5))
bars = plt.bar(
    range(len(proposer_df)),
    proposer_df["blocks_proposed"],
    color=colors,
    edgecolor="black",
    width=0.8
)
total_blocks = proposer_df["blocks_proposed"].sum()
max_height = proposer_df["blocks_proposed"].max()
for i, b in enumerate(bars):
    pct = 100 * proposer_df["blocks_proposed"].iloc[i] / total_blocks if total_blocks else 0
    if proposer_df["blocks_proposed"].iloc[i] > 0:
        plt.text(
            b.get_x() + b.get_width()/2, b.get_height() + 0.08 * max_height,
            f"{pct:.0f}%",
            ha="center", va="bottom", fontsize=10
        )
plt.ylim(0, max_height * 1.32)
plt.xticks(
    range(len(proposer_df)),
    [addr[:5] + "…" + addr[-3:] for addr in proposer_df["address"]],
    rotation=45, ha="right", fontsize=8
)
plt.ylabel("Blocks Proposed")
plt.xlabel("Proposer (sorted by Voting Power)")
plt.title(
    "Proposer Frequency vs Voting Power\n(Red: Top 5 by Voting Power; Label = % of blocks proposed)",
    fontsize=13
)
plt.legend(
    handles=[
        plt.matplotlib.patches.Patch(color="#d7263d", label="Top Voting Power Validators"),
        plt.matplotlib.patches.Patch(color="#2584a5", label="Other Validators"),
    ],
    loc="upper right"
)
plt.tight_layout()
plt.savefig(f"{plots_dir}/plot6_proposer_rotation_by_vp.pdf")
plt.savefig(f"{plots_dir}/plot6_proposer_rotation_by_vp.png")
plt.close()

# Plot 7
if "Commit Time" in df.columns:
    df["Commit Time"] = pd.to_datetime(df["Commit Time"], format="%H:%M:%S.%f", errors="coerce")

    step_cols = ["Propose → Prevote", "Prevote → Precommit", "Precommit → Commit"]
    filtered = df[["Commit Time"] + step_cols].copy()
    for c in step_cols:
        filtered[c] = pd.to_numeric(filtered[c], errors="coerce")

    melted = filtered.melt(
        id_vars=["Commit Time"],
        value_vars=step_cols,
        var_name="Consensus Step",
        value_name="Duration (ms)"
    )
    melted = melted.dropna(subset=["Commit Time"])
    melted = melted.sort_values("Commit Time")

    smoothed = (
        melted
        .set_index("Commit Time")
        .groupby("Consensus Step")["Duration (ms)"]
        .rolling("30s")
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=melted,
        x="Commit Time", y="Duration (ms)",
        hue="Consensus Step",
        marker="o", markersize=3,
        alpha=0.3, linewidth=0.8,
        legend=False
    )
    sns.lineplot(
        data=smoothed,
        x="Commit Time", y="Duration (ms)",
        hue="Consensus Step",
        linewidth=2.5,
        hue_order=step_cols,
    )
    plt.xlabel("Time")
    plt.ylabel("Duration (ms)")
    plt.title("Consensus Step Durations Over Time")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M:%S"))
    plt.legend(
        loc="upper right",
        title="Consensus Step",
        fontsize="11",
        title_fontsize="13"
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    plt.xticks(rotation=45)
    finalize_plot(
        xlabel=None,
        ylabel=None,
        title=None,
        filename="plot7_step_durations_vs_time",
        legend_loc=None
    )

# Plot 8
if "Avg Mempool Wait Time" in mempool_df_active.columns:
    plt.figure(figsize=(10, 5))
    avg_wait = mempool_df_active[["Timestamp", "Avg Mempool Wait Time"]].dropna()
    q98 = avg_wait["Avg Mempool Wait Time"].quantile(0.98)
    normal = avg_wait[avg_wait["Avg Mempool Wait Time"] <= q98]
    outliers = avg_wait[avg_wait["Avg Mempool Wait Time"] > q98]
    plt.plot(
        avg_wait["Timestamp"], 
        avg_wait["Avg Mempool Wait Time"], 
        marker="o", markersize=2, linewidth=1, color="tab:blue"
    )
    plt.scatter(
        outliers["Timestamp"], 
        outliers["Avg Mempool Wait Time"], 
        color="red", s=30, zorder=5)
    finalize_plot("Mempool Timestamp", "Average Mempool Wait Time (ms)", "Average Mempool Wait Time Over Time", "plot8_avg_wait_time_over_time")

# Plot 9
if "Max Mempool Wait Time" in mempool_df_active.columns:
    plt.figure(figsize=(10, 5))
    max_wait = mempool_df_active[["Timestamp", "Max Mempool Wait Time"]].dropna()
    q98 = max_wait["Max Mempool Wait Time"].quantile(0.98)
    normal = max_wait[max_wait["Max Mempool Wait Time"] <= q98]
    outliers = max_wait[max_wait["Max Mempool Wait Time"] > q98]
    plt.plot(max_wait["Timestamp"], max_wait["Max Mempool Wait Time"], marker="o", markersize=2, linewidth=1, color="tab:blue")
    plt.scatter(outliers["Timestamp"], outliers["Max Mempool Wait Time"], color="red", s=30, zorder=5)
    finalize_plot("Mempool Timestamp", "Max Mempool Wait Time (ms)", "Max Mempool Wait Time Over Time", "plot9_max_wait_time_over_time")

print(f"All plots saved in ./{plots_dir}/")

