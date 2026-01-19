import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean and prepare the data
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Domestic'] = pd.to_numeric(df['Domestic'].str.replace(',', ''), errors='coerce')
df['International (non-CIS)'] = pd.to_numeric(df['International (non-CIS)'].str.replace(',', ''), errors='coerce')
df['CIS'] = pd.to_numeric(df['CIS'].str.replace(',', ''), errors='coerce')

# Drop rows where any of the required columns are missing
df = df.dropna(subset=['Domestic', 'International (non-CIS)', 'CIS'])

# Set the year as index for plotting
df = df[df['Year'].between(2000, 2013)]
df = df.sort_values('Year')

# Create the stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.25
years = df['Year']
domestic = df['Domestic']
int_non_cis = df['International (non-CIS)']
cis = df['CIS']

x = range(len(years))
plt.bar(x, domestic, width=bar_width, label='Domestic', color='skyblue')
plt.bar([p + bar_width for p in x], int_non_cis, width=bar_width, label='International (non-CIS)', color='lightgreen')
plt.bar([p + 2*bar_width for p in x], cis, width=bar_width, label='CIS', color='salmon')

plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('Trends in Domestic, International (non-CIS), and CIS Passenger Counts (2000–2013)')
plt.xticks([p + bar_width for p in x], years, rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()