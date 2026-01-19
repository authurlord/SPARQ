import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert the 'Period' column to a proper index for plotting
df['Period'] = df['Period'].str.replace('-', ' to ', regex=False)

# Prepare data for plotting
births = df['Live births per year'].astype(float)
deaths = df['Deaths per year'].astype(float)
periods = df['Period']

# Create area chart
plt.figure(figsize=(12, 6))
plt.stackplot(periods, births, deaths, labels=['Live Births', 'Deaths'], alpha=0.8)
plt.title('Number of Births and Deaths Over Time Periods')
plt.xlabel('Time Period')
plt.ylabel('Number of Events')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()