import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter out the total row
df_filtered = df[df['Rank'] != 'Total']

# Convert to numeric for plotting
df_filtered[['Gold', 'Silver', 'Bronze']] = df_filtered[['Gold', 'Silver', 'Bronze']].apply(pd.to_numeric, errors='coerce')

# Plot stacked bar chart
plt.figure(figsize=(12, 8))
bars = df_filtered.plot(x='Nation', y=['Gold', 'Silver', 'Bronze'], kind='bar', stacked=True, color=['gold', 'silver', 'lightcoral'])
plt.title('Number of Gold, Silver, and Bronze Medals by Country')
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.xticks(rotation=45)
plt.legend(title='Medal Type')
plt.tight_layout()
plt.show()