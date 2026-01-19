import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the total row
df_filtered = df[df['Rank'] != 'Total']

# Prepare data for plotting
data = df_filtered[['Nation', 'Gold', 'Silver', 'Bronze']]
data.set_index('Nation', inplace=True)

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
data.plot(kind='bar', stacked=True)
plt.title('Number of Gold, Silver, and Bronze Medals by Country')
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()