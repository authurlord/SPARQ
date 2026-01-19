import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Exclude the 'total' column and keep only regional columns
regions = df.columns[1:-1]  # 'mobiles' to 'wales'
song_data = df[regions].astype(int)

# Calculate total sales per region
total_per_region = song_data.sum(axis=0)

# Calculate percentage of each song's sales per region
percentage_data = (song_data.T / total_per_region) * 100

# Plotting the percentage bar chart
plt.figure(figsize=(12, 6))
x = range(len(df['song']))
width = 0.8

for i, region in enumerate(regions):
    plt.bar([j + i * width / len(regions) for j in x], percentage_data[region], width=width/len(regions), label=region)

plt.xlabel('Songs')
plt.ylabel('Percentage of Total Sales (%)')
plt.title('Percentage of Total Sales by Song and Region')
plt.xticks([j + width/2 for j in x], df['song'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()