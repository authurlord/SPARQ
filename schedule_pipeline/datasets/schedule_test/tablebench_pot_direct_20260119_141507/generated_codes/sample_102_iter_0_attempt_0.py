import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Clean the Period column to extract years (e.g., '1950-1955' -> '1950')
df['Period'] = df['Period'].str.split('-').str[0]

# Convert live births and deaths to numeric
df['Live births per year'] = pd.to_numeric(df['Live births per year'].str.replace(' ', ''), errors='coerce')
df['Deaths per year'] = pd.to_numeric(df['Deaths per year'].str.replace(' ', ''), errors='coerce')

# Create the area chart
plt.figure(figsize=(12, 6))
plt.fill_between(df['Period'], df['Live births per year'], label='Live Births', alpha=0.7)
plt.fill_between(df['Period'], df['Deaths per year'], label='Deaths', alpha=0.7)
plt.title('Number of Births and Deaths Over Time Periods')
plt.xlabel('Time Period')
plt.ylabel('Number of Events')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()