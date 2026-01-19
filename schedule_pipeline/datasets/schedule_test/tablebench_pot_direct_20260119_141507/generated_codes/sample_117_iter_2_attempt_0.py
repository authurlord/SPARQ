import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert necessary columns to numeric, handling missing values
df['Domestic'] = pd.to_numeric(df['Domestic'].str.replace(',', ''), errors='coerce')
df['International (non-CIS)'] = pd.to_numeric(df['International (non-CIS)'].str.replace(',', ''), errors='coerce')
df['CIS'] = pd.to_numeric(df['CIS'].str.replace(',', ''), errors='coerce')

# Replace None or NaN with 0
df['International (non-CIS)'] = df['International (non-CIS)'].fillna(0)
df['CIS'] = df['CIS'].fillna(0)

# Ensure Year is numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Filter only years from 2000 to 2013
df = df[df['Year'].between(2000, 2013)]

# Prepare data for plotting
data = df[['Year', 'Domestic', 'International (non-CIS)', 'CIS']].copy()

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.2
years = data['Year'].values

# Create bars
domestic_bars = plt.bar(years - bar_width, data['Domestic'], bar_width, label='Domestic', color='skyblue')
non_cis_bars = plt.bar(years, data['International (non-CIS)'], bar_width, label='International (non-CIS)', color='lightgreen')
cis_bars = plt.bar(years + bar_width, data['CIS'], bar_width, label='CIS', color='salmon')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('Trends in Domestic, International (non-CIS), and CIS Passenger Counts (2000–2013)')
plt.xticks(years, [str(year) for year in years], rotation=45)
plt.legend()

# Improve layout
plt.tight_layout()

# Show plot
plt.show()

# Final Answer is not a value but a chart, so we just print confirmation
Final Answer: chart