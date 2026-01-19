import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and filter data: remove rows with DNF or DSQ (invalid times)
df_clean = df.dropna(subset=['Time']).copy()
df_clean['Time'] = df_clean['Time'].str.replace('[:.]', '.', regex=False)

# Convert time to seconds
def time_to_seconds(time_str):
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds

# Baseline time (first place)
baseline_time = time_to_seconds(df_clean.iloc[0]['Time'])

# Compute differences from baseline
df_clean['time_seconds'] = df_clean['Time'].apply(time_to_seconds)
df_clean['diff_from_baseline'] = df_clean['time_seconds'] - baseline_time

# Drop the first row (rank 1) since we are showing differences from it
# We want to show the changes from the baseline to each subsequent nation
# So we start at baseline, then add each difference
# We exclude the first row because its difference is zero

# Prepare the list of changes (differences) for the waterfall chart
changes = df_clean.iloc[1:].diff_from_baseline.tolist()
labels = df_clean.iloc[1:]['Nation'].tolist()

# Create the waterfall chart
plt.figure(figsize=(12, 6))
plt.barh([0], [0], color='lightblue', label='Baseline')
for i, (change, label) in enumerate(zip(changes, labels)):
    plt.barh([i], [change], color='orange' if change > 0 else 'red', label='Change' if i == 0 else "")

# Better approach: Use a proper waterfall plot
# Instead, we'll create a simple horizontal bar chart with cumulative sum
# But waterfall chart should show cumulative changes

# Rebuild the waterfall-like sequence
cumulative = [baseline_time]
for i in range(1, len(df_clean)):
    cumulative.append(cumulative[-1] + df_clean.iloc[i]['diff_from_baseline'])

# Now create a waterfall chart using cumulative values
plt.figure(figsize=(12, 6))
x = range(len(cumulative))
plt.plot(x, cumulative, marker='o', linestyle='-', color='blue', linewidth=2)
plt.title('Waterfall Chart of Completion Times Relative to First Place')
plt.xlabel('Rank')
plt.ylabel('Time (seconds)')
plt.grid(True, axis='y')
plt.xticks(x, [f'{row["Rank"]}' for _, row in df_clean.iterrows()], rotation=45)
plt.tight_layout()
plt.show()