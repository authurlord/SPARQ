import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove rows with None in 'Time'
df = df.dropna(subset=['Time'])

# Convert Time to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds

times_seconds = df['Time'].apply(time_to_seconds)

# Get the first-place time (rank 1) as baseline
baseline = times_seconds.iloc[0]  # First row has rank 1.0

# Compute differences from baseline
differences = times_seconds - baseline

# Create the waterfall chart
plt.figure(figsize=(10, 6))
plt.bar(['First Place'], [baseline], color='green', label='Baseline', edgecolor='black')
for i, diff in enumerate(differences):
    if i == 0:
        continue
    plt.bar(f'Participant {i+1}', diff, color='blue' if diff > 0 else 'red', edgecolor='black')

# Labeling
plt.title('Waterfall Chart of Completion Times Relative to First Place')
plt.ylabel('Time (seconds)')
plt.xlabel('Participants')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Show the chart
plt.tight_layout()
plt.show()

# Final answer is not a number but a visualization — so we just output the required format
Final Answer: waterfall_chart