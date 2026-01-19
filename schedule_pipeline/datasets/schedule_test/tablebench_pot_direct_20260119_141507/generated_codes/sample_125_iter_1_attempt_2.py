import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean data: drop rows where 'Time' is missing or DNF/DSQ
df_clean = df.dropna(subset=['Time'])
df_clean = df_clean[df_clean['Time'].str.contains('DNF') == False]
df_clean = df_clean[df_clean['Time'].str.contains('DSQ') == False]

# Convert time string to seconds
def time_to_seconds(time_str):
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds

# Extract baseline time (first place, rank 1)
baseline_time = time_to_seconds(df_clean.iloc[0]['Time'])

# Compute differences from baseline
df_clean['time_seconds'] = df_clean['Time'].apply(time_to_seconds)
df_clean['diff_from_baseline'] = df_clean['time_seconds'] - baseline_time

# Sort by rank to maintain order
df_clean = df_clean.sort_values(by='Rank').reset_index(drop=True)

# Prepare data for waterfall chart
times = df_clean['diff_from_baseline']
labels = df_clean['Nation']

# Create waterfall chart
plt.figure(figsize=(12, 6))
plt.bar(labels, times, color=['skyblue' if x >= 0 else 'coral' for x in times], edgecolor='black')
plt.axhline(y=0, color='black', linewidth=1)
plt.title('Waterfall Chart: Completion Times Relative to First-Place Finisher')
plt.ylabel('Time Difference (seconds)')
plt.xlabel('Nation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Final answer is not a number but a chart — so we just output the required format
Final Answer: waterfall_chart