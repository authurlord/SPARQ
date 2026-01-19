import pandas as pd
import matplotlib.pyplot as plt

# Parse the table data directly from JSON
data = [
    [' H', ' K\t8\t34\t330\t7500\t0.7\t1%\t10\t30000\tYes... 30000 in the visible band; maxiumum baseline 330-m'],
    [' No imaging on a single baseline instrument. CLOSED 2009.', None]
]

columns = ['Interferometer and observing mode', 'Waveband Limiting magnitude Minimum baseline (m)\\n(un-projected) Maximum baseline (m) Approx. no. visibility measurements per year\\n(measurements per night x nights used per year) Max ratio of no. phase / no. amplitude measurements\\n(measure of imaging performance', 'Accuracy of amplitude2 measurements Accuracy of phase measurements\\n(milli-radians) Number of spectral channels\\n(max in use simultaneously) Comments']

df = pd.DataFrame(data, columns=columns)

# Clean the data: extract minimum and maximum baseline from the second column
# Split the second column by spaces and extract numbers
def parse_baseline(row):
    if pd.isna(row) or row is None:
        return None, None
    parts = row.split('\t')
    if len(parts) < 4:
        return None, None
    try:
        min_baseline = int(parts[2])  # 34
        max_baseline = int(parts[3])  # 330
        return min_baseline, max_baseline
    except ValueError:
        return None, None

df[['min_baseline', 'max_baseline']] = df['Waveband Limiting magnitude Minimum baseline (m)\\n(un-projected) Maximum baseline (m) Approx. no. visibility measurements per year\\n(measurements per night x nights used per year) Max ratio of no. phase / no. amplitude measurements\\n(measure of imaging performance', 'Accuracy of amplitude2 measurements Accuracy of phase measurements\\n(milli-radians) Number of spectral channels\\n(max in use simultaneously) Comments'].apply(parse_baseline, axis=1).apply(pd.Series)

# Drop rows where baselines are invalid
df = df.dropna(subset=['min_baseline', 'max_baseline'])

# Filter out the row with "No imaging..." since it's not a valid mode
df = df[df['Interferometer and observing mode'] != ' No imaging on a single baseline instrument. CLOSED 2009.']

# Prepare data for plotting
x = df['Interferometer and observing mode']
min_baselines = df['min_baseline']
max_baselines = df['max_baseline']

# Create grouped bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(x))

plt.bar(index, min_baselines, bar_width, label='Minimum Baseline (m)', color='skyblue')
plt.bar([i + bar_width for i in index], max_baselines, bar_width, label='Maximum Baseline (m)', color='salmon')

plt.xlabel('Interferometer and Observing Mode')
plt.ylabel('Baseline (m)')
plt.title('Maximum and Minimum Baselines for Different Interferometer and Observing Modes')
plt.xticks([i + bar_width / 2 for i in index], x, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()