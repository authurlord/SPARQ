import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Convert the 'Year' column to integer for proper plotting
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Extract relevant columns (only the international non-CIS and CIS, and domestic)
# Note: The 'International (non-CIS)' and 'CIS' columns have missing values in 2013
# We will drop rows where any of these values are NaN for plotting

# Filter out rows with missing values in domestic, international (non-CIS), or CIS
df_clean = df.dropna(subset=['Domestic', 'International (non-CIS)', 'CIS'])

# Convert the relevant columns to numeric
df_clean['Domestic'] = pd.to_numeric(df_clean['Domestic'], errors='coerce')
df_clean['International (non-CIS)'] = pd.to_numeric(df_clean['International (non-CIS)'], errors='coerce')
df_clean['CIS'] = pd.to_numeric(df_clean['CIS'], errors='coerce')

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.25
index = range(len(df_clean))

# Plot each series
bars1 = plt.bar([i - bar_width for i in index], df_clean['Domestic'], bar_width, label='Domestic')
bars2 = plt.bar([i for i in index], df_clean['International (non-CIS)'], bar_width, label='International (non-CIS)')
bars3 = plt.bar([i + bar_width for i in index], df_clean['CIS'], bar_width, label='CIS')

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Passenger Count')
plt.title('Trends in Domestic, International (non-CIS), and CIS Passenger Counts (2000–2013)')
plt.xticks(index, df_clean['Year'])
plt.legend()

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()