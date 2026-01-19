import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling None/NaN
df['Length [km]'] = pd.to_numeric(df['Length [km]'], errors='coerce')
df['Drainage basin area [km2]'] = pd.to_numeric(df['Drainage basin area [km2]'], errors='coerce')

# Drop rows with missing values in either column
df_clean = df.dropna(subset=['Length [km]', 'Drainage basin area [km2]'])

# Prepare data for plotting
names = df_clean['Name']
lengths = df_clean['Length [km]']
basins = df_clean['Drainage basin area [km2]']

# Create figure and axis
plt.figure(figsize=(14, 8))

# Create grouped bar chart
x = range(len(names))
width = 0.35

plt.bar([i - width/2 for i in x], lengths, width, label='Length [km]', color='skyblue')
plt.bar([i + width/2 for i in x], basins, width, label='Drainage basin area [km2]', color='lightcoral')

# Labels and title
plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in x], names, rotation=90)

# Legend
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()