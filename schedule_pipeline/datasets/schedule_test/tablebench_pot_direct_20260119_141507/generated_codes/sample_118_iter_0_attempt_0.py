import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the total row (last row with 'Total' in Nation)
df_filtered = df[df['Nation'] != 'Total']

# Pivot the data to have countries as index and medals as columns
pivot_data = df_filtered[['Nation', 'Gold', 'Silver', 'Bronze']].set_index('Nation')

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
pivot_data.plot(kind='bar', stacked=True, color=['gold', 'silver', 'lightcoral'])
plt.title('Number of Gold, Silver, and Bronze Medals by Country')
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.xticks(rotation=45)
plt.legend(title='Medal Type')
plt.tight_layout()
plt.show()