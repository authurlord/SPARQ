import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Remove the 'Total' row
df_filtered = df[df['Rank'] != 'Total']

# Convert the medal columns to integers
df_filtered[['Gold', 'Silver', 'Bronze']] = df_filtered[['Gold', 'Silver', 'Bronze']].astype(int)

# Create a pivot table to have one row per country with gold, silver, bronze
pivot_data = df_filtered[['Nation', 'Gold', 'Silver', 'Bronze']]
pivot_data.set_index('Nation', inplace=True)

# Plotting: Use a grouped bar chart for gold, silver, and bronze
plt.figure(figsize=(12, 8))
pivot_data.plot(kind='bar', width=0.6, color=['gold', 'silver', 'lightcoral'], edgecolor='black')
plt.title('Number of Gold, Silver, and Bronze Medals by Country')
plt.xlabel('Country')
plt.ylabel('Number of Medals')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Medal Type')
plt.tight_layout()
plt.show()

print(f"Final Answer: Chart plotted successfully")