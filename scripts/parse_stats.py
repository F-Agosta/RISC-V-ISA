import os
import sys
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  Helper: parse a gem5 stats.txt into instruction mix + health       #
# ------------------------------------------------------------------ #

def parse_stats(stats_file):
    instruction_mix = {}
    health_metrics  = {}

    if not os.path.exists(stats_file):
        print(f"[WARN] stats file not found: {stats_file}")
        return instruction_mix, health_metrics

    with open(stats_file, "r") as f:
        for line in f:
            if line.startswith("simInsts") or \
               line.startswith("board.processor.cores.core.cpi"):
                parts = line.split()
                name  = parts[0].split('.')[-1]
                value = float(parts[1])
                health_metrics[name] = value

            elif "issuedInstType_0::" in line and "total" not in line:
                parts     = line.split()
                inst_name = parts[0].split("::")[-1]
                count     = int(parts[1])
                if count > 0:
                    instruction_mix[inst_name] = count

    return instruction_mix, health_metrics


# ------------------------------------------------------------------ #
#  Helper: save bar chart                                              #
# ------------------------------------------------------------------ #

def save_bar_chart(mix, title, output_path, color="skyblue"):
    if not mix:
        print(f"[WARN] no instruction data for: {title}")
        return

    sorted_mix = dict(sorted(mix.items(), key=lambda x: x[1], reverse=True))
    labels = list(sorted_mix.keys())
    values = list(sorted_mix.values())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("Instruction Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar_label(bars, padding=3, fontsize=7)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"  Chart saved  : {output_path}")


# ------------------------------------------------------------------ #
#  Helper: save instruction mix as text                               #
# ------------------------------------------------------------------ #

def save_text(mix, health, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base = output_path.replace("_mix", "_health").replace("_plot", "")
    health_path = base.replace("_mix", "_health")

    # instruction mix
    with open(output_path, "w") as f:
        for inst, cnt in sorted(mix.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{inst}: {cnt}\n")
    print(f"  Mix saved    : {output_path}")

    # health metrics
    with open(health_path, "w") as f:
        for k, v in health.items():
            f.write(f"{k}: {v}\n")
    print(f"  Health saved : {health_path}")


# ------------------------------------------------------------------ #
#  CLI arguments                                                       #
# ------------------------------------------------------------------ #

if len(sys.argv) > 1:
    N_FEATURES, N_DATAPOINTS = sys.argv[1].split(',')
else:
    N_FEATURES, N_DATAPOINTS = "0", "0"

tag = f"{N_FEATURES}_{N_DATAPOINTS}"
os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------------ #
#  Parse all three pipelines                                           #
# ------------------------------------------------------------------ #

print("\n=== Parsing baseline (fp32) ===")
mix_base, health_base = parse_stats("data/m5out_baseline/stats.txt")
# Normalize 100 vectors to 2 vectors (divide by 50)
mix_base = {k: v / 50 for k, v in mix_base.items()}
if 'simInsts' in health_base:
    health_base['simInsts'] /= 50
print(f"  Health: {health_base}")

print("\n=== Parsing fp16 custom ISA ===")
mix_fp16, health_fp16 = parse_stats("data/m5out_fp16/stats.txt")
# Normalize 100 vectors to 2 vectors (divide by 50)
mix_fp16 = {k: v / 50 for k, v in mix_fp16.items()}
if 'simInsts' in health_fp16:
    health_fp16['simInsts'] /= 50
print(f"  Health: {health_fp16}")

print("\n=== Parsing HDC-RFF ===")
mix_hdc, health_hdc = parse_stats("data/m5out_hdc/stats.txt")
print(f"  Health: {health_hdc}")

print("\n=== Parsing HDC-RFF Custom ISA ===")
mix_hdc_custom, health_hdc_custom = parse_stats("data/m5out_hdc_custom/stats.txt")
print(f"  Health: {health_hdc_custom}")

# ------------------------------------------------------------------ #
#  Per-pipeline charts & text files                                    #
# ------------------------------------------------------------------ #

print("\n=== Saving individual charts ===")

save_bar_chart(mix_base, "RISC-V Instruction Mix — RBF Kernel (fp32)",
               f"results/instruction_mix_plot_{tag}.png", color="steelblue")
save_text(mix_base, health_base, f"results/instruction_mix_{tag}.txt")

save_bar_chart(mix_fp16, "RISC-V Instruction Mix — RBF Kernel (fp16 SIMD)",
               f"results/fp16_instruction_mix_plot_{tag}.png", color="darkorange")
save_text(mix_fp16, health_fp16, f"results/fp16_instruction_mix_{tag}.txt")

save_bar_chart(mix_hdc, "RISC-V Instruction Mix — HDC-RFF",
               f"results/hdc_instruction_mix_plot_{tag}.png", color="mediumseagreen")
save_text(mix_hdc, health_hdc, f"results/hdc_instruction_mix_{tag}.txt")

save_bar_chart(mix_hdc_custom, "RISC-V Instruction Mix — HDC-RFF Custom ISA",
               f"results/hdc_custom_instruction_mix_plot_{tag}.png", color="crimson")
save_text(mix_hdc_custom, health_hdc_custom, f"results/hdc_custom_instruction_mix_{tag}.txt")

# ------------------------------------------------------------------ #
#  Three-way comparison chart                                          #
# ------------------------------------------------------------------ #

print("\n=== Saving three-way comparison chart ===")

# Union of all instruction types that appear in at least one pipeline
all_insts = sorted(
    set(mix_base) | set(mix_fp16) | set(mix_hdc) | set(mix_hdc_custom),
    key=lambda k: max(mix_base.get(k, 0), mix_fp16.get(k, 0), mix_hdc.get(k, 0), mix_hdc_custom.get(k, 0)),
    reverse=True
)

if all_insts:
    x     = range(len(all_insts))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(16, len(all_insts) * 1.3), 8))

    bars_base = ax.bar([i - 1.5*width for i in x],
                       [mix_base.get(k, 0) for k in all_insts],
                       width, label="RBF fp32 (baseline)", color="steelblue",
                       edgecolor="black")
    bars_fp16 = ax.bar([i - 0.5*width for i in x],
                       [mix_fp16.get(k, 0) for k in all_insts],
                       width, label="RBF fp16 (custom ISA)", color="darkorange",
                       edgecolor="black")
    bars_hdc  = ax.bar([i + 0.5*width for i in x],
                       [mix_hdc.get(k, 0) for k in all_insts],
                       width, label="HDC-RFF", color="mediumseagreen",
                       edgecolor="black")
    bars_hdc_custom = ax.bar([i + 1.5*width for i in x],
                             [mix_hdc_custom.get(k, 0) for k in all_insts],
                             width, label="HDC-RFF Custom ISA", color="crimson",
                             edgecolor="black")

    ax.set_title("Four-Way Instruction Mix Comparison")
    ax.set_ylabel("Instruction Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_insts, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out = f"results/comparison_plot_{tag}.png"
    plt.savefig(out)
    plt.close()
    print(f"  Chart saved  : {out}")
else:
    print("  [WARN] No instruction data available for comparison chart.")
