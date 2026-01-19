import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and prepare data
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Domestic'] = pd.to_numeric(df['Domestic'].str.replace(',', ''), errors='coerce')
df['International (non-CIS)'] = pd.to_numeric(df['International (non-CIS)'].str.replace(',', ''), errors='coerce')
df['CIS'] = pd.to_numeric(df['CIS'].str.replace(',', ''), errors='coerce')

# Replace None or NaN with 0
df['Domestic'] = df['Domestic'].fillna(0)
df['International (non-CIS)'] = df['International (non-CIS)'].fillna(0)
df['CIS'] = df['CIS'].fillna(0)

# Filter only years from 2000 to 2013
df_filtered = df[df['Year'].between(2000, 2013)]

# Pivot for plotting
data = df_filtered[['Year', 'Domestic', 'International (non-CIS)', 'CIS']]
data.set_index('Year', inplace=True)

# Create stacked bar chart
plt.figure(figsize=(12, 6))
data.plot(kind='bar', stacked=True)
plt.title('Trends in Domestic, International (non-CIS), and CIS Passenger Counts (2000-2013)')
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()