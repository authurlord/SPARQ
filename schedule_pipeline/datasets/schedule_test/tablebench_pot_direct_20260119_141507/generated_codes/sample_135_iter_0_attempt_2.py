import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the region columns to numeric (since they are strings with numbers)
region_cols = ['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']
df[region_cols] = df[region_cols].apply(pd.to_numeric, errors='coerce')

# Calculate the percentage of each region's sales relative to the total for each song
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df['percentages'] = df[region_cols].apply(lambda x: x / df['total'], axis=1)

# Create a pivot table to get percentages per song and region
pivot = df[region_cols].div(df['total'], axis=0)

# Plot the percentage bar chart
fig, ax = plt.subplots(figsize=(12, 8))
pivot.plot(kind='bar', ax=ax, width=0.8, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'orange'])

# Customize the plot
ax.set_title('Percentage of Total Sales for Each Song by Region')
ax.set_xlabel('Song')
ax.set_ylabel('Percentage of Total Sales')
ax.set_xticklabels(pivot.index, rotation=45)
plt.tight_layout()

# Show the plot
plt.show()