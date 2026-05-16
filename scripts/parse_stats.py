import os
import matplotlib.pyplot as plt

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
os.mkdir("results", exist_ok=True)
plt.savefig("results/instruction_mix_plot.png")
print("Visual saved to results/instruction_mix_plot.png")
