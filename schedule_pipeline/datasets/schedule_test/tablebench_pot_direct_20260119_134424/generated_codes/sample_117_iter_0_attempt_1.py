import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove spaces and convert to integers
df['Domestic'] = df['Domestic'].str.replace(' ', '').astype(int)
df['International (non-CIS)'] = df['International (non-CIS)'].str.replace(' ', '').astype(int)
df['CIS'] = df['CIS'].str.replace(' ', '').astype(int)

# Extract the years and data for the stacked bar chart
years = df['Year']
domestic = df['Domestic']
non_cis = df['International (non-CIS)']
cis = df['CIS']

# Create the stacked bar chart
plt.figure(figsize=(12, 6))
plt.bar(years, domestic, label='Domestic', color='skyblue')
plt.bar(years, non_cis, bottom=domestic, label='International (non-CIS)', color='lightgreen')
plt.bar(years, cis, bottom=domestic + non_cis, label='CIS', color='lightcoral')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('Trends in Domestic, International (non-CIS), and CIS Passenger Count (2000–2013)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()