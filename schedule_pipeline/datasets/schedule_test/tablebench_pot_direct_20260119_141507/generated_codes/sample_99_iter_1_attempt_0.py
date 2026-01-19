import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert length and drainage basin area to numeric, replacing None with NaN
df['Length [km]'] = pd.to_numeric(df['Length [km]'], errors='coerce')
df['Drainage basin area [km2]'] = pd.to_numeric(df['Drainage basin area [km2]'], errors='coerce')

# Drop rows where both values are NaN (in case of missing data)
df_clean = df.dropna(subset=['Length [km]', 'Drainage basin area [km2]'])

# Prepare data for plotting
rivers = df_clean['Name']
lengths = df_clean['Length [km]']
basins = df_clean['Drainage basin area [km2]']

# Create a figure and axis
plt.figure(figsize=(12, 6))
x = range(len(rivers))
width = 0.35

# Create two bars per river
plt.bar([i - width/2 for i in x], lengths, width, label='Length [km]', color='skyblue')
plt.bar([i + width/2 for i in x], basins, width, label='Drainage basin area [km2]', color='lightcoral')

# Labels and title
plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in x], rivers, rotation=90)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()