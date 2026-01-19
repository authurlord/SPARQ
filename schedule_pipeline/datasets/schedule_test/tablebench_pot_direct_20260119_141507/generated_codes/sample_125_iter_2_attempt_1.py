import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean data: remove rows with DNF or DSQ
df_clean = df.dropna(subset=['Time']).copy()
df_clean = df_clean[df_clean['Time'] != 'DNF']
df_clean = df_clean[df_clean['Time'] != 'DSQ']

# Filter only those with Heat = 1 or 2, but keep all since we are focusing on time
# Convert time to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    h, m_s = time_str.split(':')
    m, s = m_s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s)

# Apply conversion
df_clean['time_seconds'] = df_clean['Time'].apply(time_to_seconds)

# Get baseline: first place time (rank 1)
baseline_time = time_to_seconds('2:59.30')

# Compute differences from baseline
df_clean['diff_from_baseline'] = df_clean['time_seconds'] - baseline_time

# Sort by rank to maintain order
df_clean = df_clean.sort_values(by='Rank').reset_index(drop=True)

# Extract the list of differences (excluding the first row which is baseline)
differences = df_clean['diff_from_baseline'].tolist()

# Create the waterfall chart
plt.figure(figsize=(12, 6))
plt.bar(['Baseline'], [0], color='lightblue', label='Baseline')
# Add bars for each participant's difference
for i, diff in enumerate(differences):
    if i == 0:
        continue
    # We start from baseline and add each difference
    # So we use cumulative effect
    # But for waterfall, we show each step as a vertical bar
    # We'll use the actual difference as the value
    # First bar is baseline, then each next is a step
    # So we just plot the differences in sequence
    plt.bar(f'Participant {i+1}', diff, bottom=0, color='skyblue' if diff >= 0 else 'salmon')

# Labeling
plt.axhline(0, color='black', linewidth=1)
plt.title("Waterfall Chart of Completion Times Relative to First Place")
plt.xlabel("Participants")
plt.ylabel("Time Difference (seconds)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()