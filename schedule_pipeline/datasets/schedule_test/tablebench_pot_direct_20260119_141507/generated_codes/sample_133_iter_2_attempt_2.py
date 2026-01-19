import pandas as pd
import matplotlib.pyplot as plt

# Parse the table data directly from JSON
data = [
    [' H', ' K\t8\t34\t330\t7500\t0.7\t1%\t10\t30000\tYes... 30000 in the visible band; maxiumum baseline 330-m'],
    [' No imaging on a single baseline instrument. CLOSED 2009.', None]
]

columns = ['Interferometer and observing mode', 'Waveband Limiting magnitude Minimum baseline (m)\\n(un-projected) Maximum baseline (m) Approx. no. visibility measurements per year\\n(measurements per night x nights used per year) Max ratio of no. phase / no. amplitude measurements\\n(measure of imaging performance', 'Accuracy of amplitude2 measurements Accuracy of phase measurements\\n(milli-radians) Number of spectral channels\\n(max in use simultaneously) Comments']

df = pd.DataFrame(data, columns=columns)

# Clean and parse the second column to extract min and max baseline
df['min_baseline'] = None
df['max_baseline'] = None

# Only process the first row since the second row has None
if not df.iloc[0][1] is None:
    # Split the string by tab and extract the 3rd and 4th values (index 2 and 3)
    parts = df.iloc[0][1].split('\t')
    if len(parts) >= 4:
        min_baseline = int(parts[2])  # index 2: '34'
        max_baseline = int(parts[3])  # index 3: '330'
        df.loc[0, 'min_baseline'] = min_baseline
        df.loc[0, 'max_baseline'] = max_baseline

# Filter out invalid rows
df_filtered = df.dropna(subset=['min_baseline'])

# Prepare data for plotting
modes = df_filtered['Interferometer and observing mode']
min_baselines = df_filtered['min_baseline']
max_baselines = df_filtered['max_baseline']

# Create grouped bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(modes))

plt.bar(index, min_baselines, bar_width, label='Minimum Baseline (m)', color='skyblue')
plt.bar([i + bar_width for i in index], max_baselines, bar_width, label='Maximum Baseline (m)', color='salmon')

plt.xlabel('Interferometer and Observing Mode')
plt.ylabel('Baseline (m)')
plt.title('Maximum and Minimum Baselines for Different Interferometer and Observing Modes')
plt.xticks([i + bar_width / 2 for i in index], modes, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()