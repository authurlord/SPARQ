import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Exclude the 'total' column and convert other columns to numeric
regions = df.columns[1:-1]  # Exclude 'song' and 'total'
df[regions] = df[regions].astype(int)

# Calculate total sales across all songs and regions
total_sales = df[regions].sum().sum()

# Calculate percentage for each region per song
percentages = df[regions].div(total_sales) * 100

# Plot the percentage bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.1
index = range(len(df))

for i, region in enumerate(regions):
    ax.bar([j + i * bar_width for j in index], percentages[region], bar_width, label=region)

ax.set_xlabel('Songs')
ax.set_ylabel('Percentage of Total Sales (%)')
ax.set_title('Percentage of Total Sales by Song and Region')
ax.set_xticks([j + bar_width * (len(regions) - 1) / 2 for j in index])
ax.set_xticklabels(df['song'], rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

# Final Answer: No numerical output required; chart is generated.