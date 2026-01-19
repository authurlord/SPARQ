import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: drop rows where 'Time' is None or DNF/DSQ
df_clean = df.dropna(subset=['Time'])
df_clean = df_clean[df_clean['Time'] != 'DNF']
df_clean = df_clean[df_clean['Time'] != 'DSQ']

# Convert time strings to seconds
def time_to_seconds(time_str):
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds

# Extract baseline time (first place)
baseline_time = time_to_seconds('2:59.30')

# Compute differences from baseline
df_clean['time_seconds'] = df_clean['Time'].apply(time_to_seconds)
df_clean['diff_from_baseline'] = df_clean['time_seconds'] - baseline_time

# Sort by rank (to maintain order)
df_clean['Rank'] = df_clean['Rank'].astype(str).str.strip()
df_clean = df_clean.dropna(subset=['Rank']).reset_index(drop=True)

# Extract the list of differences for the waterfall chart
differences = df_clean['diff_from_baseline'].tolist()

# Create the waterfall chart
plt.figure(figsize=(12, 6))
plt.bar(range(len(differences)), differences, color=['blue' if d > 0 else 'red' for d in differences], alpha=0.8)
plt.axhline(0, color='black', linewidth=1)
plt.title("Completion Times Relative to First-Place Finisher (Baseline: 2:59.30)")
plt.ylabel("Time Difference (seconds)")
plt.xlabel("Participants")
plt.xticks(range(len(differences)), [f"Rank {row['Rank']}" for _, row in df_clean.iterrows()], rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()