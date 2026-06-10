import os
import matplotlib.pyplot as plt
import sys

stats_file = "data/m5out_baseline/stats.txt"

instruction_mix = {}
health_metrics = {}

with open(stats_file, "r") as f:
    for line in f:
        if line.startswith("simInsts") or line.startswith("board.processor.cores.core.cpi"):
            parts = line.split() 
            name = parts[0].split('.')[-1] 
            value = float(parts[1])
            health_metrics[name] = value
            
        elif "issuedInstType_0::" in line and not "total" in line:
            parts = line.split()
            inst_name = parts[0].split("::")[1] 
            count = int(parts[1])
            
            if count > 0:
                instruction_mix[inst_name] = count

print(f"Simulation Health: {health_metrics}")

# --- VISUALS ---

sorted_inst = dict(sorted(instruction_mix.items(), key=lambda item: item[1], reverse=True))


labels = list(sorted_inst.keys())
values = list(sorted_inst.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color='skyblue', edgecolor='black')
plt.title('RISC-V Instruction Mix (RBF Kernel)')
plt.ylabel('Instruction Count')
plt.xticks(rotation=45) # Tilt the labels so they fit
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.bar_label(plt.bar(labels, values), padding=3) # Add labels on top of bars

plt.tight_layout()
os.makedirs("results", exist_ok=True)
if (len(sys.argv) > 1):
    N_FEATURES, N_DATAPOINTS = sys.argv[1].split(',')
else:
    N_FEATURES, N_DATAPOINTS = 0, 0

# Save graph
plt.savefig(f"results/instruction_mix_plot_{N_FEATURES}_{N_DATAPOINTS}.png")
print(f"Graph saved to results/instruction_mix_plot_{N_FEATURES}_{N_DATAPOINTS}.png")

# Save health metrics data
with open(f"results/health_metrics_{N_FEATURES}_{N_DATAPOINTS}.txt", "w") as f:
    for key, value in health_metrics.items():
        f.write(f"{key}: {value}\n")
print(f"Health metrics saved to results/health_metrics_{N_FEATURES}_{N_DATAPOINTS}.txt")

# Save instruction mix data
with open(f"results/instruction_mix_{N_FEATURES}_{N_DATAPOINTS}.txt", "w") as f:
    for inst, count in sorted_inst.items():
        f.write(f"{inst}: {count}\n")
print(f"Instruction mix saved to results/instruction_mix_{N_FEATURES}_{N_DATAPOINTS}.txt")
