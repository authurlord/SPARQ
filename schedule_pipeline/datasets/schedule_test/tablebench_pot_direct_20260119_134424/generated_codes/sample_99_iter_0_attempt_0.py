import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert columns to numeric, handling any non-numeric values
df['Length [km]'] = pd.to_numeric(df['Length [km]'], errors='coerce')
df['Drainage basin area [km2]'] = pd.to_numeric(df['Drainage basin area [km2]'], errors='coerce')

# Drop rows with missing values in either column
df.dropna(subset=['Length [km]', 'Drainage basin area [km2]'], inplace=True)

# Set up the bar chart
plt.figure(figsize=(14, 7))
bar_width = 0.35
indices = range(len(df))

# Create bars
plt.bar(indices, df['Length [km]'], bar_width, label='Length [km]', color='skyblue')
plt.bar([i + bar_width for i in indices], df['Drainage basin area [km2]'], bar_width, label='Drainage Basin Area [km2]', color='lightgreen')

# Add labels and title
plt.xlabel('Rivers')
plt.ylabel('Values')
plt.title('River Length and Drainage Basin Area')
plt.xticks([i + bar_width / 2 for i in indices], df['Name'], rotation=45, ha='right')
plt.legend()

# Show the chart
plt.tight_layout()
plt.show()