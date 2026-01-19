import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the 'Total' row
df_filtered = df[df['Nation'] != 'Total']

# Pivot the data to have each medal type as a separate column
pivot_df = df_filtered[['Nation', 'Gold', 'Silver', 'Bronze']].copy()

# Plot grouped bar chart
plt.figure(figsize=(12, 7))
bar_width = 0.2
index = range(len(pivot_df))

# Create bars for each medal type
plt.bar([i - bar_width for i in index], pivot_df['Gold'], bar_width, label='Gold', color='gold')
plt.bar([i          for i in index], pivot_df['Silver'], bar_width, label='Silver', color='silver')
plt.bar([i + bar_width for i in index], pivot_df['Bronze'], bar_width, label='Bronze', color='lightcoral')

# Labels and title
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.title('Number of Gold, Silver, and Bronze Medals by Country')
plt.xticks(index, pivot_df['Nation'], rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()