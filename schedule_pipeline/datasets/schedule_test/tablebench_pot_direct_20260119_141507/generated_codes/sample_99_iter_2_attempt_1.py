import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Select only relevant columns
df_filtered = df[['Name', 'Length [km]', 'Drainage basin area [km2]']]

# Convert to numeric, handling missing values
df_filtered['Length [km]'] = pd.to_numeric(df_filtered['Length [km]'], errors='coerce')
df_filtered['Drainage basin area [km2]'] = pd.to_numeric(df_filtered['Drainage basin area [km2]'], errors='coerce')

# Drop rows with missing values in either column
df_filtered = df_filtered.dropna(subset=['Length [km]', 'Drainage basin area [km2]'])

# Prepare data for plotting
names = df_filtered['Name']
lengths = df_filtered['Length [km]']
basins = df_filtered['Drainage basin area [km2]']

# Create figure and axis
plt.figure(figsize=(14, 8))

# Position of bars on x-axis
x = range(len(names))
width = 0.35

# Create bars
bars1 = plt.bar([i - width/2 for i in x], lengths, width, label='Length [km]', color='skyblue')
bars2 = plt.bar([i + width/2 for i in x], basins, width, label='Drainage basin area [km2]', color='lightcoral')

# Add labels and title
plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in x], names, rotation=90)

# Add legend
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()