import pandas as pd

# Load the dataset
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric (handle any potential issues)
df['2008 gdp per capita (usd) a'] = pd.to_numeric(df['2008 gdp per capita (usd) a'], errors='coerce')
df['exports (usd mn) 2011'] = pd.to_numeric(df['exports (usd mn) 2011'], errors='coerce')

# Remove rows with NaN due to conversion issues
df = df.dropna(subset=['2008 gdp per capita (usd) a', 'exports (usd mn) 2011'])

# Calculate mean and std for the two key variables
gdp_per_capita_mean = df['2008 gdp per capita (usd) a'].mean()
gdp_per_capita_std = df['2008 gdp per capita (usd) a'].std()

exports_mean = df['exports (usd mn) 2011'].mean()
exports_std = df['exports (usd mn) 2011'].std()

# Identify outliers using 2 standard deviations rule
outliers = []

for idx, row in df.iterrows():
    gdp_val = row['2008 gdp per capita (usd) a']
    export_val = row['exports (usd mn) 2011']
    district = row['district']
    
    # Check if GDP per capita is extreme
    if abs(gdp_val - gdp_per_capita_mean) > 2 * gdp_per_capita_std:
        outliers.append(district)
    
    # Check if exports are extreme
    if abs(export_val - exports_mean) > 2 * exports_std:
        if district not in outliers:
            outliers.append(district)

# Remove duplicates and sort
outliers = list(set(outliers))
outliers.sort()

print(f"Final Answer: {', '.join(outliers)}")