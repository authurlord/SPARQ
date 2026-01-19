import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Define the region columns
regions = ['mobiles', 'northern ireland', 'northern england', 'scotland', 'southern england', 'wales']

# Calculate total sales across all songs and regions
total_sales = df['total'].sum()

# Calculate percentage for each region per song
for region in regions:
    df[f'{region}_percentage'] = (df[region] / total_sales) * 100

# Prepare data for plotting
songs = df['song']
region_percentages = df[[f'{region}_percentage' for region in regions]]

# Plot the grouped bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.1
index = range(len(songs))

for i, region in enumerate(regions):
    plt.bar([x + i * bar_width for x in index], df[f'{region}_percentage'], bar_width, label=region)

plt.xlabel('Song')
plt.ylabel('Percentage of Total Sales (%)')
plt.title('Percentage of Total Sales by Song and Region')
plt.xticks([x + bar_width * (len(regions) - 1) / 2 for x in index], songs, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()