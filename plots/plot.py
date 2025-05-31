import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# Set style for better visualization
plt.style.use('seaborn-v0_8-deep')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 8  # Base font size
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files for generation 1
pattern = os.path.join(script_dir, 'gen1_run*_sac_ppo_SAC_1.csv')
csv_files = glob.glob(pattern)

if not csv_files:
    print(f"No CSV files found matching pattern: {pattern}")
    exit(1)

# Create figure with column-appropriate size (assuming 3.5 inch column width)
plt.figure(figsize=(3.5, 4))

# Create main plot area
main_ax = plt.gca()

# Prepare data for mean and std calculation
all_steps = None
all_values = []
run_data = {}

# First pass: collect all data
for csv_file in sorted(csv_files):
    run_num = int(os.path.basename(csv_file).split('run')[1].split('_')[0])
    df = pd.read_csv(csv_file)
    df['Step'] = df['Step'] / 1000  # Convert to thousands
    
    if all_steps is None:
        all_steps = df['Step'].values
    run_data[run_num] = df['Value'].values
    all_values.extend(df['Value'].values)

# Calculate mean and std
values_array = np.array([run_data[run] for run in sorted(run_data.keys())])
mean_values = np.mean(values_array, axis=0)
std_values = np.std(values_array, axis=0)

# Plot confidence interval
main_ax.fill_between(all_steps, 
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.2,
                    color='gray',
                    label='±1 std dev')

# Plot mean line
main_ax.plot(all_steps, mean_values,
            color='black',
            linewidth=2,
            linestyle='--',
            label='Mean',
            zorder=10)

# Plot individual runs
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
for i, run_num in enumerate(sorted(run_data.keys())):
    values = run_data[run_num]
    
    # Plot the run
    main_ax.plot(all_steps, values,
                label=f'Run {run_num}',
                color=colors[i],
                marker='o',
                markersize=4,
                linewidth=1,
                alpha=0.8,
                zorder=5)

# Find best and worst runs
final_values = {run: values[-1] for run, values in run_data.items()}
best_run = max(final_values.items(), key=lambda x: x[1])[0]
worst_run = min(final_values.items(), key=lambda x: x[1])[0]

# Add annotations for best and worst runs
main_ax.annotate(f'Best ({best_run})',
                xy=(all_steps[-1], run_data[best_run][-1]),
                xytext=(5, 5),
                textcoords='offset points',
                ha='left',
                va='bottom',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

main_ax.annotate(f'Worst ({worst_run})',
                xy=(all_steps[-1], run_data[worst_run][-1]),
                xytext=(5, -5),
                textcoords='offset points',
                ha='left',
                va='top',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Customize the plot
main_ax.set_title('Mean Evaluation Reward', pad=10)
main_ax.set_xlabel('Training Steps (thousands)')
main_ax.set_ylabel('Reward')
main_ax.grid(True, linestyle='--', alpha=0.7)

# Set axis limits with margin
margin = (max(all_values) - min(all_values)) * 0.1
main_ax.set_ylim(min(all_values) - margin, max(all_values) + margin)
main_ax.set_xlim(min(all_steps) - 0.5, max(all_steps) + 0.5)

# Add minor gridlines
main_ax.grid(True, which='minor', linestyle=':', alpha=0.4)
main_ax.minorticks_on()

# Add legend below the plot
legend = main_ax.legend(
    ncol=3,  # Arrange legend in 3 columns
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    fontsize=7,
    frameon=True,
    handlelength=1,
    handletextpad=0.4,
    columnspacing=1,
    borderpad=0.2
)

# Adjust layout
plt.tight_layout()

# Save the plot with absolute path
output_file = os.path.join(script_dir, 'generation1_comparison_paper.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nPaper-formatted plot saved as '{output_file}'")
