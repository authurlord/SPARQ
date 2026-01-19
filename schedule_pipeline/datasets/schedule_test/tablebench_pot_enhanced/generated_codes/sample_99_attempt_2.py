import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter out rows where drainage basin area is missing
df_filtered = df.dropna(subset=['Drainage basin area [km2]'])

# Convert the columns to numeric
df_filtered['Length [km]'] = pd.to_numeric(df_filtered['Length [km]'], errors='coerce')
df_filtered['Drainage basin area [km2]'] = pd.to_numeric(df_filtered['Drainage basin area [km2]'], errors='coerce')

# Drop any remaining NaN values
df_filtered = df_filtered.dropna()

# Prepare data for plotting
rivers = df_filtered['Name']
lengths = df_filtered['Length [km]']
basin_areas = df_filtered['Drainage basin area [km2]']

# Create figure and axis
plt.figure(figsize=(14, 8))

# Create grouped bar chart
bar_width = 0.35
index = range(len(rivers))
bars1 = plt.bar([i - bar_width/2 for i in index], lengths, bar_width, label='Length [km]', color='skyblue')
bars2 = plt.bar([i + bar_width/2 for i in index], basin_areas, bar_width, label='Drainage basin area [km2]', color='lightcoral')

# Labels and title
plt.xlabel('River Name')
plt.ylabel('Value')
plt.title('Length of Rivers and Their Drainage Basin Areas')
plt.xticks([i for i in index], rivers, rotation=90)
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()

print(f"Final Answer: Bar chart created successfully")