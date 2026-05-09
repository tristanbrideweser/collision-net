import re
import matplotlib.pyplot as plt

# Files to parse
log_files = ['slurm-454625.out', 'slurm-454628.out']
epochs, losses, accs = [], [], []

for log_file in log_files:
    with open(log_file, 'r') as f:
        for line in f:
            # Matches: Epoch 13 | Val Loss: 0.0950 | Acc: 96.05%
            match = re.search(r'Epoch (\d+) \| Val Loss: ([\d.]+) \| Acc: ([\d.]+)%', line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accs.append(float(match.group(3)))

# Sort by epoch to handle the resume/relay chain
data = sorted(zip(epochs, losses, accs))
epochs, losses, accs = zip(*data)

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Loss
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss', color='tab:red')
ax1.plot(epochs, losses, color='tab:red', label='Val Loss', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:red')

# Plot Accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:blue')
ax2.plot(epochs, accs, color='tab:blue', label='Val Acc', linewidth=2)
ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% Goal')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('CollisionNet Training Progress (Model A: Expert)')
fig.tight_layout()
plt.savefig('collisionnet_performance.png', dpi=300)
print("✅ Plot saved as collisionnet_performance.png")
