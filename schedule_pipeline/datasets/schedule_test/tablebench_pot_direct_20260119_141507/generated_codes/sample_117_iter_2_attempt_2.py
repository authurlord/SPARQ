import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling missing values
df['Domestic'] = pd.to_numeric(df['Domestic'].str.replace(',', ''), errors='coerce')
df['International (non-CIS)'] = pd.to_numeric(df['International (non-CIS)'].str.replace(',', ''), errors='coerce')
df['CIS'] = pd.to_numeric(df['CIS'], errors='coerce')

# Drop rows where any of the required columns are NaN
df_clean = df.dropna(subset=['Domestic', 'International (non-CIS)', 'CIS'])

# Ensure the year column is correct and sorted
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
df_clean = df_clean[df_clean['Year'].notna()]

# Sort by Year
df_clean = df_clean.sort_values('Year').reset_index(drop=True)

# Prepare data for plotting
years = df_clean['Year'].values
domestic = df_clean['Domestic'].values
non_cis_intl = df_clean['International (non-CIS)'].values
cis = df_clean['CIS'].values

# Create the stacked bar chart
plt.figure(figsize=(12, 7))
bars = plt.bar(years, domestic, label='Domestic', color='skyblue')
plt.bar(years, non_cis_intl, bottom=domestic, label='International (non-CIS)', color='lightgreen')
plt.bar(years, cis, bottom=domestic + non_cis_intl, label='CIS', color='salmon')

plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('Trends in Domestic, International (non-CIS), and CIS Passenger Count (2000–2013)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()