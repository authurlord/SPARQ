import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Remove the 'Total' row
df = df[df['Rank'] != 'Total']

# Convert medal columns to numeric (in case of string issues)
df[['Gold', 'Silver', 'Bronze']] = df[['Gold', 'Silver', 'Bronze']].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN after conversion
df = df.dropna(subset=['Gold', 'Silver', 'Bronze'])

# Create a grouped bar chart for Gold, Silver, and Bronze medals
plt.figure(figsize=(12, 8))
bar_width = 0.2
index = range(len(df))

# Plot bars
bars1 = plt.bar([i - bar_width for i in index], df['Gold'], bar_width, label='Gold', color='gold')
bars2 = plt.bar([i for i in index], df['Silver'], bar_width, label='Silver', color='silver')
bars3 = plt.bar([i + bar_width for i in index], df['Bronze'], bar_width, label='Bronze', color='bronze')

# Labels and title
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.title('Number of Gold, Silver, and Bronze Medals by Country')
plt.xticks([i for i in index], df['Nation'], rotation=45)

# Add legend
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()