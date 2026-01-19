import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter out invalid rows (where first column is None)
valid_data = df.dropna(subset=['Interferometer and observing mode'])

# Extract relevant columns
modes = valid_data['Interferometer and observing mode']
min_baselines = valid_data['Minimum baseline (m)\n(un-projected)'].fillna(0)
max_baselines = valid_data['Maximum baseline (m)'].fillna(0)

# Since only one valid mode exists, we create a simple bar chart
if len(modes) == 0:
    print("No valid data to plot.")
else:
    # Prepare data for plotting
    mode_name = modes.iloc[0]
    min_baseline = min_baselines.iloc[0]
    max_baseline = max_baselines.iloc[0]
    
    # Create a grouped bar chart with two bars per mode
    fig, ax = plt.subplots(figsize=(8, 6))
    positions = [0, 1]
    labels = ['Minimum Baseline', 'Maximum Baseline']
    values = [min_baseline, max_baseline]
    
    ax.bar(positions, values, color=['skyblue', 'lightcoral'], edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(f'Maximum and Minimum Baselines for {mode_name}')
    ax.set_ylabel('Baseline (m)')
    
    plt.tight_layout()
    plt.show()