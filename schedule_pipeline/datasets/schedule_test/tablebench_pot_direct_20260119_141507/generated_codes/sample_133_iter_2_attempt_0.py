import pandas as pd
import matplotlib.pyplot as plt

# Parse the table data directly from JSON
data = [
    [' H', ' K\t8\t34\t330\t7500\t0.7\t1%\t10\t30000\tYes... 30000 in the visible band; maxiumum baseline 330-m'],
    [' No imaging on a single baseline instrument. CLOSED 2009.', None]
]

# Prepare the DataFrame
df = pd.DataFrame(data, columns=['Interferometer and observing mode', 'Details'])

# Extract the values from the second column (Details) and split by tab
df['Details'] = df['Details'].astype(str).str.replace('\t', ' ', regex=False)
df['Details'] = df['Details'].str.split(' ', n=1, expand=True)

# Extract minimum and maximum baseline values from the formatted string
# The format is: "K 8 34 330 ..." so we need to extract 34 (min) and 330 (max)
def parse_baseline(row):
    if pd.isna(row) or row == '':
        return None, None
    parts = row.split()
    # Find indices of min and max baseline
    # We assume the sequence: [mode, 8, 34, 330, ...]
    # So 34 is min baseline, 330 is max baseline
    try:
        min_baseline = int(parts[2])
        max_baseline = int(parts[3])
        return min_baseline, max_baseline
    except (IndexError, ValueError):
        return None, None

df[['min_baseline', 'max_baseline']] = df['Details'].apply(parse_baseline, result_type='expand')

# Filter out rows where either baseline is missing
df = df.dropna(subset=['min_baseline', 'max_baseline'])

# Create grouped bar chart
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(df))
plt.bar([i - bar_width/2 for i in index], df['min_baseline'], bar_width, label='Minimum Baseline (m)')
plt.bar([i + bar_width/2 for i in index], df['max_baseline'], bar_width, label='Maximum Baseline (m)')

plt.xlabel('Interferometer and Observing Mode')
plt.ylabel('Baseline (m)')
plt.title('Maximum and Minimum Baselines for Different Interferometer and Observing Modes')
plt.xticks([i for i in index], df['Interferometer and observing mode'], rotation=45)
plt.legend()

plt.tight_layout()
plt.show()