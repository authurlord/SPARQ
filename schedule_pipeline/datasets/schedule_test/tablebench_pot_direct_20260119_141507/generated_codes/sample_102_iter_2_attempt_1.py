import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert 'Live births per year' and 'Deaths per year' to numeric by removing spaces
df['Live births per year'] = df['Live births per year'].str.replace(' ', '').astype(int)
df['Deaths per year'] = df['Deaths per year'].str.replace(' ', '').astype(int)

# Set the period as index for plotting
df.set_index('Period', inplace=True)

# Create the area chart
plt.figure(figsize=(12, 6))
plt.fill_between(df.index, df['Live births per year'], label='Live Births', alpha=0.7)
plt.fill_between(df.index, df['Deaths per year'], label='Deaths', alpha=0.7)

plt.title('Number of Births and Deaths Over Time Periods')
plt.xlabel('Time Period')
plt.ylabel('Number per Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()