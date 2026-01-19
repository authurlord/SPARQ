import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean data: filter out rows with invalid or missing times
df_clean = df.dropna(subset=['Time'])

# Remove rows with DNF or DSQ
df_clean = df_clean[df_clean['Time'].str.contains('DNF|DSQ') == False]

# Convert time to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds

# Apply conversion
df_clean['time_seconds'] = df_clean['Time'].apply(time_to_seconds)

# Get the baseline: first-place finisher (rank 1.0)
baseline_time = time_to_seconds('2:59.30')

# Extract only valid times and nations (excluding rank and notes)
valid_times = df_clean['time_seconds'].dropna()
nations = df_clean['Nation'].dropna()

# Compute differences from baseline
differences = valid_times - baseline_time

# Create a list of labels (nations) and differences
labels = [f"{nation}" for nation in nations]
values = differences.tolist()

# Create waterfall chart
plt.figure(figsize=(12, 6))
plt.bar(labels, values, color=['skyblue' if x >= 0 else 'coral' for x in values], edgecolor='black')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.title("Waterfall Chart of Completion Times Relative to First-Place Finisher")
plt.ylabel("Time Difference (seconds)")
plt.xlabel("Nation")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()