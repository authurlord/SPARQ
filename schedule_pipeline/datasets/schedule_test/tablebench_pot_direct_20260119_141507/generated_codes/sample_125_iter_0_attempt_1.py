import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and filter out invalid entries (DNF, DSQ)
df = df.dropna(subset=['Time'])
df = df[df['Time'].str.contains('DNF') == False]
df = df[df['Time'].str.contains('DSQ') == False]

# Convert 'Time' to seconds
def time_to_seconds(time_str):
    if ':' not in time_str:
        return 0
    mins, secs = time_str.split(':')
    return int(mins) * 60 + float(secs)

times_seconds = df['Time'].apply(time_to_seconds)
baseline_time = time_to_seconds('2:59.30')  # First-place time

# Compute differences from baseline
differences = times_seconds - baseline_time

# Create the waterfall chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(differences)), differences, color=['green' if diff > 0 else 'red' for diff in differences], alpha=0.8)
plt.axhline(y=0, color='black', linewidth=1)
plt.title('Waterfall Chart of Completion Times vs First-Place Finisher')
plt.xlabel('Participant Rank')
plt.ylabel('Time Difference (seconds)')
plt.xticks(range(len(differences)), df['Rank'].astype(str), rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()