import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area (km 2 )'] = pd.to_numeric(df['area (km 2 )'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Calculate Z-scores for area and population
from scipy.stats import zscore
df['area_zscore'] = zscore(df['area (km 2 )'])
df['population_zscore'] = zscore(df['population'])

# Define threshold for outlier (e.g., |Z| > 3)
outliers = df[(df['area_zscore'].abs() > 3) | (df['population_zscore'].abs() > 3)]

# Extract place names of outliers
outlier_places = outliers['place'].tolist()
print(f"Final Answer: {', '.join(outlier_places)}")