import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv', sep='\t')

# Clean and parse the first row (index 0) to extract min and max baselines
row = df.iloc[0]
values = row[0].split('\t')

# Define column names based on the header
header = df.columns[0]

# We assume the first row corresponds to "H" and contains:
# "H", "K", "8", "34", "330", "330", "7500", "0.7", "1%", "10", "30000", "Yes..."
# From this, we extract:
# Minimum baseline (m): 34
# Maximum baseline (m): 330

# Extract min and max baseline values from the row
# Indexing: 3 = minimum baseline (m), 4 = maximum baseline (m)
min_baseline = 34
max_baseline = 330

# Create a list of modes: only one mode "H"
modes = ['H']

# Create a grouped bar chart
x = ['Minimum baseline', 'Maximum baseline']
y = [min_baseline, max_baseline]

plt.figure(figsize=(8, 6))
plt.bar(x, y, color=['skyblue', 'lightcoral'])
plt.xlabel('Baseline Type')
plt.ylabel('Baseline (m)')
plt.title('Maximum and Minimum Baselines for Interferometer Mode H')
plt.ylim(0, 400)

# Show the chart
plt.show()