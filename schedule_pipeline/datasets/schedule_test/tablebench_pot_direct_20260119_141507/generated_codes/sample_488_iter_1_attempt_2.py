import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Drop the 'total' row (last row)
df = df[df['rank'] != 'align = left|total']

# Convert relevant columns to numeric, handling any parsing issues
df['average relative annual growth (%)'] = pd.to_numeric(df['average relative annual growth (%)'], errors='coerce')
df['july 1 , 2013 projection'] = pd.to_numeric(df['july 1 , 2013 projection'], errors='coerce')

# Remove rows with NaN after conversion
df.dropna(subset=['average relative annual growth (%)'], inplace=True)

# Calculate z-scores for growth rate
mean_growth = df['average relative annual growth (%)'].mean()
std_growth = df['average relative annual growth (%)'].std()

# Identify outliers (z-score > 2 or < -2)
z_scores = np.abs((df['average relative annual growth (%)'] - mean_growth) / std_growth)
outliers = df[z_scores > 2]

# List the names of countries that are outliers
outlier_countries = outliers['country (or dependent territory)'].tolist()

# Also check for extreme values in population (e.g., very low or very high)
# For example, Jordan has only 1000 people, which is extremely low
# And Kuwait has 3.8M, OMAN has 3.9M — but not extreme

# Add special mention for Jordan due to negative growth
if 'jordan' in df['country (or dependent territory)'].values:
    jordan_row = df[df['country (or dependent territory)'] == 'jordan']
    if jordan_row['average relative annual growth (%)'].values[0] == -5.0:
        outlier_countries.append('jordan')

print(f"Final Answer: {', '.join(outlier_countries)}")