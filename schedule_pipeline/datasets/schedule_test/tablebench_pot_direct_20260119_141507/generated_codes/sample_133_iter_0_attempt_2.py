import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter out the row with None in the first column
df = df.dropna(subset=['Interferometer and observing mode'])

# Extract the relevant columns
df = df[['Interferometer and observing mode', 'Minimum baseline (m)\n(un-projected)', 'Maximum baseline (m)']]

# Convert to numeric, handling any formatting issues
df['Minimum baseline (m)\n(un-projected)'] = pd.to_numeric(df['Minimum baseline (m)\n(un-projected)'], errors='coerce')
df['Maximum baseline (m)'] = pd.to_numeric(df['Maximum baseline (m)'], errors='coerce')

# Drop any rows with NaN after conversion
df = df.dropna()

# Prepare data for plotting
modes = df['Interferometer and observing mode']
min_baselines = df['Minimum baseline (m)\n(un-projected)']
max_baselines = df['Maximum baseline (m)']

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35
index = range(len(modes))

bars1 = ax.bar([i - bar_width/2 for i in index], min_baselines, bar_width, label='Minimum baseline (m)')
bars2 = ax.bar([i + bar_width/2 for i in index], max_baselines, bar_width, label='Maximum baseline (m)')

ax.set_xlabel('Interferometer and Observing Mode')
ax.set_ylabel('Baseline (m)')
ax.set_title('Maximum and Minimum Baselines by Interferometer and Observing Mode')
ax.set_xticks([i for i in index])
ax.set_xticklabels(modes)
ax.legend()

plt.tight_layout()
plt.show()