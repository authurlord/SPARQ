import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Drop rows where either 'Length [km]' or 'Drainage basin area [km2]' is NaN
df_clean = df.dropna(subset=['Length [km]', 'Drainage basin area [km2]'])

# Convert the columns to numeric
df_clean['Length [km]'] = pd.to_numeric(df_clean['Length [km]'], errors='coerce')
df_clean['Drainage basin area [km2]'] = pd.to_numeric(df_clean['Drainage basin area [km2]'], errors='coerce')

# Drop any remaining NaN values
df_clean = df_clean.dropna()

# Prepare data for plotting
rivers = df_clean['Name']
lengths = df_clean['Length [km]']
basins = df_clean['Drainage basin area [km2]']

# Create a figure and axis
plt.figure(figsize=(14, 8))

# Create grouped bar chart
x = range(len(rivers))
width = 0.35

plt.bar([i - width/2 for i in x], lengths, width, label='Length [km]', color='skyblue')
plt.bar([i + width/2 for i in x], basins, width, label='Drainage basin area [km2]', color='lightcoral')

plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in x], rivers, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()